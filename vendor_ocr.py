import cv2
import numpy as np
from pathlib import Path
import easyocr
import re
import pandas as pd
from difflib import SequenceMatcher

# Fix SSL certificate verification error for EasyOCR model downloads on Windows
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request._create_default_https_context = ssl._create_unverified_context

# Minimal OCR-based vendor detection module used by other scripts.

_reader = None
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def preprocess_channels(img):
    channels = []
    for i in range(3):
        channels.append(img[:, :, i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    channels.append(gray)
    channels.append(255 - gray)
    processed = []
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    for ch in channels:
        try:
            ch2 = clahe.apply(ch)
            ch2 = cv2.filter2D(ch2, -1, kernel)
        except Exception:
            ch2 = ch
        processed.append(ch2)
    return processed

def merge_rects(rects, overlapThresh=0.3):
    if len(rects) == 0:
        return []
    rects = np.array(rects)
    x1 = rects[:,0]; y1 = rects[:,1]
    x2 = rects[:,0] + rects[:,2]; y2 = rects[:,1] + rects[:,3]
    areas = rects[:,2] * rects[:,3]
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs)-1]
        for pos in range(len(idxs)-1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i]); yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i]); yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            overlap = (w * h) / areas[i]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return rects[pick].tolist()

def detect_text_regions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create()
    except Exception:
        return []
    mser.setMinArea(30)
    mser.setMaxArea(8000)
    regions, _ = mser.detectRegions(gray)
    rects = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1,1,2))
        if w > 5 and h > 5:
            rects.append((x,y,w,h))
    return merge_rects(rects)

def ocr_image(img):
    final_texts = set()
    scales = [0.5, 1.0, 1.5, 2.0]
    for scale in scales:
        h = int(img.shape[0] * scale)
        w = int(img.shape[1] * scale)
        if h <= 0 or w <= 0:
            continue
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        channels = preprocess_channels(resized)
        for ch in channels:
            try:
                results = get_reader().readtext(ch)
            except Exception:
                results = []
            for item in results:
                # easyocr may return (bbox, text, conf) or (bbox, text)
                text = item[1] if len(item) > 1 else str(item)
                text = text.strip()
                if not text:
                    continue
                if len(re.findall(r'[A-Za-z]', text)) < 3:
                    continue
                final_texts.add(text)
    rects = detect_text_regions(img)
    for (x,y,w,h) in rects:
        roi = img[y:y+h, x:x+w]
        channels = preprocess_channels(roi)
        for ch in channels:
            try:
                results = get_reader().readtext(ch)
            except Exception:
                results = []
            for item in results:
                text = item[1] if len(item) > 1 else str(item)
                text = text.strip()
                if not text:
                    continue
                if len(re.findall(r'[A-Za-z]', text)) < 3:
                    continue
                final_texts.add(text)
    return " \n ".join(final_texts) if final_texts else ""

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = s.replace('0', 'o').replace('1', 'l')
    s = s.replace('\n', ' ')
    s = re.sub(r'[^a-z0-9\s\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# A compact vendor list; can be loaded from external source if desired
VENDORS = [
 'Adtran','Advantech Co. Ltd','Aerohive Switches','airpro','Alaxala','alcatel-lucent enterprise',
 'Allied Telesis','Alpha Networks','AMG Systems','Antaira Technologies','Araknis','Arista Networks','Aruba',
 'Avaya','Belden','Brocade','Cisco','D-Link','Dell Technologies','Edgecore Networks','Extreme Networks',
 'Fortinet','H3C','Huawei','Juniper Networks','Lenovo','LINKOH','Linksys','Mikrotik','NETGEAR','Nokia Networks',
 'QNAP Systems','Quanta Cloud Technology (QCT)','RUCKUS','Ruijie Networks','Schneider Electric','TP-link','Ubiquiti Networks','ZTE','Zyxel'
]

ALIASES = {}
try:
    alias_path = Path(__file__).parent / 'alias_switches.xlsx'
    if alias_path.exists():
        df = pd.read_excel(alias_path, header=None)
        for _, row in df.iterrows():
            cells = [str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip()]
            if not cells:
                continue
            canonical = cells[0]
            for a in cells:
                key = normalize_text(a)
                if key:
                    ALIASES.setdefault(canonical, set()).add(key)
except Exception:
    ALIASES = {}

def best_vendor_match(text: str, vendors=VENDORS) -> str:
    if not text:
        return None
    txt = normalize_text(text)
    if not txt:
        return None
    norm_vendors = [(v, normalize_text(v)) for v in vendors]
    scores = {}
    for v, nv in norm_vendors:
        s = 0.0
        if nv and nv in txt:
            s += 1.0
        t_tokens = set(txt.split())
        v_tokens = set(nv.split())
        if v_tokens:
            s += 0.6 * (len(t_tokens & v_tokens) / max(len(v_tokens), 1))
        r = SequenceMatcher(None, txt, nv).ratio()
        s += 0.5 * r
        scores[v] = s
    best = max(scores.items(), key=lambda kv: kv[1]) if scores else (None, 0)
    return best[0] if best[1] >= 0.25 else None

def classify_detected_text(detected_text: str) -> dict:
    raw_parts = re.split(r'[\n,;]+', detected_text or '')
    parts = [p.strip() for p in raw_parts if p and p.strip()]
    vendor_counts = {}
    vendor_scores = {}
    vendor_examples = {}

    vendor_aliases = {v: set([normalize_text(v)]) for v in VENDORS}
    for canonical, alias_set in ALIASES.items():
        norm_c = normalize_text(canonical)
        matched = None
        for v in VENDORS:
            if normalize_text(v) == norm_c:
                matched = v; break
        if matched:
            vendor_aliases[matched].update(alias_set)
        else:
            vendor_aliases.setdefault(canonical, set()).update(alias_set | {norm_c})

    def score_for_part_and_vendor(part_text, vendor_norm_or_aliases):
        t = normalize_text(part_text)
        if not t or not vendor_norm_or_aliases:
            return 0.0
        aliases = vendor_norm_or_aliases if isinstance(vendor_norm_or_aliases, (set, list, tuple)) else [vendor_norm_or_aliases]
        best = 0.0
        for vendor_norm in aliases:
            if not vendor_norm:
                continue
            score = 0.0
            if vendor_norm in t:
                score += 1.0
            t_tokens = set(t.split())
            v_tokens = set(vendor_norm.split())
            score += 0.6 * (len(t_tokens & v_tokens) / max(len(v_tokens), 1))
            r = SequenceMatcher(None, t, vendor_norm).ratio()
            score += 0.5 * r
            if score > best:
                best = score
        return best

    for p in parts:
        best_for_part = None
        best_score = 0.0
        candidates_to_score = [p]
        if '(' in p:
            left = p.split('(',1)[0].strip()
            if left:
                candidates_to_score.append(left)
        for sub in candidates_to_score:
            for v, aliases in vendor_aliases.items():
                s = score_for_part_and_vendor(sub, aliases)
                if s > best_score:
                    best_score = s; best_for_part = v
        if best_for_part and best_score >= 0.2:
            vendor_counts[best_for_part] = vendor_counts.get(best_for_part, 0) + 1
            vendor_scores[best_for_part] = vendor_scores.get(best_for_part, 0.0) + best_score
            vendor_examples.setdefault(best_for_part, []).append((p, round(best_score,3)))

    candidates = []
    for v in vendor_counts:
        candidates.append((v, {"count": vendor_counts[v], "score": round(vendor_scores.get(v,0.0),3), "examples": vendor_examples.get(v, [])}))
    matched = None
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda x: (x[1]["count"], x[1]["score"]), reverse=True)
        matched = candidates_sorted[0][0]
    return {"matched_vendor": matched, "candidates": candidates}

def detect_vendor(image_path):
    """Return the detected vendor name (string) or None."""
    try:
        p = Path(image_path)
        if not p.exists():
            return None
        img = cv2.imread(str(p))
        if img is None:
            return None
        text = ocr_image(img)
        if not text or text.strip() == "":
            return None
        classification = classify_detected_text(text)
        return classification.get('matched_vendor')
    except Exception:
        return None

# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path
import easyocr
from html import escape
import re
import json
from difflib import SequenceMatcher
from collections import defaultdict
import pandas as pd
import concurrent.futures
# ------------------------------- PATHS -------------------------------
IMAGES_FOLDER = Path( r"") #don't keep any path here
ALIAS_EXCEL = Path(r"F:\backend_functionality\Racktrack0912 (2)\Racktrack0912\Description_excel\excel_ocr_switch.xlsx")
OUTPUT_HTML = "ocr_vendor_clear.html"

# ------------------------------- OCR --------------------------------
_reader = None
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def preprocess_channels(img):
    """Return multiple preprocessed channels to detect text of any color/background."""
    channels = []
    # Original RGB channels
    for i in range(3):
        channels.append(img[:,:,i])
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    channels.append(gray)
    # Inverted grayscale
    channels.append(255 - gray)
    # CLAHE + sharpening
    processed = []
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    for ch in channels:
        ch = clahe.apply(ch)
        ch = cv2.filter2D(ch, -1, kernel)
        processed.append(ch)
    return processed

def detect_text_regions(img):
    """Use MSER to detect text regions for small/tiny text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(8000)
    regions, _ = mser.detectRegions(gray)
    rects = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if w > 5 and h > 5:
            rects.append((x, y, w, h))
    return merge_rects(rects)

def merge_rects(rects, overlapThresh=0.3):
    """Merge overlapping rectangles to avoid duplicate OCR."""
    if len(rects) == 0:
        return []
    rects = np.array(rects)
    x1 = rects[:,0]; y1 = rects[:,1]
    x2 = rects[:,0] + rects[:,2]; y2 = rects[:,1] + rects[:,3]
    areas = rects[:,2] * rects[:,3]
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs)-1]
        for pos in range(len(idxs)-1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i]); yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i]); yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            overlap = (w * h) / areas[i]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return rects[pick].tolist()

def ocr_image(img):
    """Combine full-image multi-scale OCR and region-based OCR."""
    final_texts = set()
    # --- Multi-scale & multi-channel OCR ---
    scales = [0.5, 1.0, 1.5, 2.0]
    for scale in scales:
        height = int(img.shape[0] * scale)
        width = int(img.shape[1] * scale)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        channels = preprocess_channels(resized)
        for ch in channels:
            results = get_reader().readtext(ch)
            for (_, text, _) in results:
                text = text.strip()
                if text:
                    # ignore OCR fragments with fewer than 3 letters
                    if len(re.findall(r'[A-Za-z]', text)) < 3:
                        continue
                    final_texts.add(text)
    # --- MSER region-based OCR ---
    rects = detect_text_regions(img)
    for (x, y, w, h) in rects:
        # NOTE: Using your original slicing as-is (no changes to existing logic)
        roi = img[y:y+h, x:x+w]
        roi_channels = preprocess_channels(roi)
        for ch in roi_channels:
            results = get_reader().readtext(ch)
            for (_, text, _) in results:
                text = text.strip()
                if text:
                    # ignore OCR fragments with fewer than 3 letters
                    if len(re.findall(r'[A-Za-z]', text)) < 3:
                        continue
                    final_texts.add(text)
    return " \n ".join(final_texts) if final_texts else "[No text detected]"

# ------------------------------ NORMALIZE ----------------------------
def normalize_text(s: str) -> str:
    """Normalize OCR text for matching: lowercase, remove non-alphanum, collapse spaces."""
    s = s or ""
    s = s.lower()
    # Replace common OCR confusions
    s = s.replace('0', 'o')
    s = s.replace('1', 'l')
    s = s.replace('\\n', ' ')
    # Remove punctuation but keep spaces
    s = re.sub(r'[^a-z0-9\s\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ------------------------------ VENDOR LIST --------------------------
VENDORS = ['Adtran','Advantech Co. Ltd','Aerohive Switches','airpro','Alaxala','alcatel-lucent enterprise',
 'Allied Telesis','Alpha Networks','AMG Systems','Antaira Technologies','Araknis','Arista Networks','Aruba',
 'Asterfusion Data Technologies','AvaLan Wireless Systems Inc','Avaya','Beckhoff Automation','Beijer Electronics',
 'beijing fibridge','Belden','Billion Electric','Brainboxes','Brocade','buffalo-technology','C-Data','Celestica',
 'CERIO Corporation','Chilinkiot','ciena','Cisco','D-Link','Datto','Dell Technologies','Delta Electronics (Agema Systems)',
 'Draytek','Edgecore Networks','Edimax Technology Co., Ltd','egenius','EtherWan','Extreme Networks','Fortinet','FS (FiberStore)',
 'H3C','HARTING','Hikvision Digital Technology','Huawei','ICP','Intellisystem Technologies','IP Infusion (OcNOS)',
 'Juniper Networks','Korenix Technology','Kyland','L-com','Lantech Communications','Lantronix','Lenovo','LINKOH','Linksys',
 'Maple Systems','Micas Networks','MICROSENS','Mikrotik','MOXA','NETGEAR','Nokia Networks','Nomadix','NoviFlow (NoviWare OS)',
 'NSGate','NVIDIA','nvt phybridge','obsidian control systems','Omnitron Systems Technology','Oracle Corporation',
 'ORing Industrial Networking','Perle Systems','Phoenix Contact (including Hirschmann)','PLANET','Proscend Communications Inc',
 'QNAP Systems','Quanta Cloud Technology (QCT)','Red Lion Controls','RS PRO','Rubytech','RUCKUS','Ruijie Networks',
 'Schneider Electric','Schweitzer Engineering Laboratories','Scomp Enterprises Private Limited','Signamax','Silicom Ltd','StarTech',
 'Supermicro','TE Connectivity','Tejas Networks','Telco','Teltonika Networks','TOTOLINK','TP-link','TrendNet','Ubiquiti Networks',
 'UfiSpace','Versa Networks','Vertiv','WAGO','Wanglink','Westermo','Xexagon','Yamaha','Yokogawa','ZTE','Zyxel']

# ------------------------------ LOAD ALIASES -------------------------
ALIASES = {}
try:
    alias_path = ALIAS_EXCEL
    if alias_path.exists():
        df = pd.read_excel(alias_path, header=None)
        for _, row in df.iterrows():
            cells = [str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip()]
            if not cells:
                continue
            canonical = cells[0]
            for a in cells:
                key = normalize_text(a)
                if key:
                    ALIASES.setdefault(canonical, set()).add(key)
except Exception:
    ALIASES = {}

# ------------------------------ VENDOR MATCH -------------------------
def best_vendor_match(text: str, vendors=VENDORS, threshold=0.6) -> str:
    if not text:
        return None
    txt = normalize_text(text)
    if not txt:
        return None
    norm_vendors = [(v, normalize_text(v)) for v in vendors]
    scores = []
    # substring boost
    for v, nv in norm_vendors:
        if nv and nv in txt:
            scores.append((v, 1.0))
    # token overlap
    txt_tokens = set(txt.split())
    for v, nv in norm_vendors:
        vendor_tokens = set(nv.split())
        if not vendor_tokens:
            continue
        intersect = txt_tokens & vendor_tokens
        token_score = len(intersect) / max(len(vendor_tokens), 1)
        if token_score > 0:
            scores.append((v, 0.6 * token_score))
    # fuzzy
    for v, nv in norm_vendors:
        if not nv:
            continue
        r = SequenceMatcher(None, txt, nv).ratio()
        if r > 0:
            scores.append((v, 0.5 * r))
    if not scores:
        return None
    agg = {}
    for v, s in scores:
        agg[v] = agg.get(v, 0.0) + s
    best_vendor = max(agg.items(), key=lambda kv: kv[1])
    if best_vendor[1] >= 0.25:
        return best_vendor[0]
    return None

def classify_detected_text(detected_text: str) -> dict:
    raw_parts = re.split(r'[\n,\n;]+', detected_text)
    parts = [p.strip() for p in raw_parts if p and p.strip()]
    vendor_counts = {}
    vendor_scores = {}
    vendor_examples = {}
    vendor_substring_counts = {}

    def score_for_part_and_vendor(part_text, vendor_norm_or_aliases):
        t = normalize_text(part_text)
        if not t or not vendor_norm_or_aliases:
            return 0.0
        aliases = vendor_norm_or_aliases if isinstance(vendor_norm_or_aliases, (set, list, tuple)) else [vendor_norm_or_aliases]
        best = 0.0
        for vendor_norm in aliases:
            if not vendor_norm:
                continue
            score = 0.0
            if vendor_norm in t:
                score += 1.0
            t_tokens = set(t.split())
            v_tokens = set(vendor_norm.split())
            score += 0.6 * (len(t_tokens & v_tokens) / max(len(v_tokens), 1))
            r = SequenceMatcher(None, t, vendor_norm).ratio()
            score += 0.5 * r
            if score > best:
                best = score
        return best

    vendor_aliases = {v: set([normalize_text(v)]) for v in VENDORS}
    for canonical, alias_set in ALIASES.items():
        norm_c = normalize_text(canonical)
        matched = None
        for v in VENDORS:
            if normalize_text(v) == norm_c:
                matched = v; break
        if matched:
            vendor_aliases[matched].update(alias_set)
        else:
            vendor_aliases.setdefault(canonical, set()).update(alias_set | {norm_c})

    for p in parts:
        candidates_to_score = [p]
        if '(' in p:
            left = p.split('(', 1)[0].strip()
            if left:
                candidates_to_score.append(left)
        best_for_part = None
        best_score = 0.0
        for sub in candidates_to_score:
            sub_norm = normalize_text(sub)
            for v, aliases in vendor_aliases.items():
                for a in aliases:
                    if a and a in sub_norm:
                        vendor_substring_counts[v] = vendor_substring_counts.get(v, 0) + 1
                        break
            for v, aliases in vendor_aliases.items():
                s = score_for_part_and_vendor(sub, aliases)
                if s > best_score:
                    best_score = s; best_for_part = v
        if best_for_part and best_score >= 0.2:
            vendor_counts[best_for_part] = vendor_counts.get(best_for_part, 0) + 1
            vendor_scores[best_for_part] = vendor_scores.get(best_for_part, 0.0) + best_score
            vendor_examples.setdefault(best_for_part, []).append((p, round(best_score, 3)))

    candidates = []
    for v in vendor_counts:
        candidates.append((v, {"count": vendor_counts[v], "score": round(vendor_scores.get(v, 0.0), 3), "examples": vendor_examples.get(v, [])}))

    # Finalize by choosing the vendor with the highest aggregated score (then count as tiebreaker).
    matched = None
    if candidates:
        # pick by score primarily, then by count
        best = max(candidates, key=lambda x: (x[1].get("score", 0.0), x[1].get("count", 0)))
        matched = best[0]
    return {"matched_vendor": matched, "candidates": candidates}

# -------------------- NEW: MODEL DETECTION (SEPARATE LOGIC) --------------------
# NOTE: This section is independent of (and does not modify) your vendor detection logic.

# Vendor-specific model regexes: prefer JSON file, fallback to defaults below
VENDOR_PATTERNS_PATH = Path(__file__).parent / "vendor_patterns.json"

# Default fallback patterns (kept from original hardcoded list)
DEFAULT_MODEL_PATTERNS = {
    "D-Link": [
        r"\bDBS-\s*2000-\s*\d+\s*MP\b",
        r"\bDES-\s*\d{3,4}[A-Z]?\b",
        r"\bDGS-\s*\d{3,4}[A-Z]?\b",
        r"\bDXS-\s*\d{3,4}[A-Z]?\b",
    ],
    "Cisco": [
        r"\bWS-\s*C\d{3,4}-\d+[A-Z]{0,3}\b",
        r"\bCatalyst\s*2960[XS]?\b",
        r"\b2960[XS]?\b",
        r"\b3750[XE]?\b",
    ],
    "Mikrotik": [
        r"\bCSS\d{3}-\d+G-\d+S\+\s*RM\b",
        r"\bCRS\d{3}-\d+G-\d+S\+\s*RM\b",
        r"\bCCR\d{3,4}-\d+[A-Z]?\b",
        r"\bRouterBOARD\b",
    ],
    "TP-link": [r"\bTL-\s*SG\d{3,4}[A-Z]?\b"],
    "NETGEAR": [r"\bGS\d{3,4}[A-Z]?\b", r"\bJGS\d{3,4}[A-Z]?\b"],
    "Zyxel": [r"\bGS\d{3,4}[A-Z]?\b", r"\bXS\d{3,4}[A-Z]?\b"],
    "Ubiquiti Networks": [r"\bUSW-\s*\d{2,3}[A-Z]?\b"],
}

def _load_model_patterns_from_json(path: Path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # Expecting a dict of vendor -> list of regex strings
                if isinstance(data, dict):
                    cleaned = {k: v for k, v in data.items() if isinstance(v, list)}
                    if cleaned:
                        return cleaned
    except Exception:
        pass
    return None

# Use JSON patterns if available, otherwise fall back
MODEL_PATTERNS = _load_model_patterns_from_json(VENDOR_PATTERNS_PATH) or DEFAULT_MODEL_PATTERNS

# Helper to compile regex pattern lists (defined early so callers can use it)
def _compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

# Optional prebuilt vendor -> image basenames JSON (no extensions)
VENDOR_IMAGES_JSON = Path(__file__).parent / "vendor_images.json"

def _load_vendor_images(path: Path):
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

VENDOR_IMAGES = _load_vendor_images(VENDOR_IMAGES_JSON)

# Build normalized lookup maps for vendor keys to handle unicode hyphens/spaces
def _normalize_vendor_key(s: str) -> str:
    if not s:
        return ""
    k = s.lower()
    k = k.replace('\u2011', '-')  # non-breaking hyphen -> normal
    k = k.replace('\u2013', '-')  # en-dash
    k = k.replace('\u2014', '-')  # em-dash
    k = re.sub(r"[^a-z0-9]+", '', k)
    return k

# normalized key -> original key for vendor images
VENDOR_IMAGES_NORM = { _normalize_vendor_key(k): k for k in VENDOR_IMAGES.keys() }

# Compile patterns and also build a normalized-key mapping for patterns
COMPILED_MODEL_PATTERNS = {v: _compile_patterns(ps) for v, ps in MODEL_PATTERNS.items()}
COMPILED_MODEL_PATTERNS_NORM = {}
for v, pats in COMPILED_MODEL_PATTERNS.items():
    nk = _normalize_vendor_key(v)
    COMPILED_MODEL_PATTERNS_NORM[nk] = pats

def _normalize_basename(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = s.replace('_', ' ')
    s = s.replace('.', ' ')
    s = re.sub(r"[^a-z0-9\s\-+]+", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s

def _score_candidate(text_norm: str, cand_norm: str) -> float:
    score = 0.0
    if cand_norm and cand_norm in text_norm:
        score += 2.0
    t_tokens = set(text_norm.split())
    c_tokens = set(cand_norm.split())
    token_overlap = len(t_tokens & c_tokens) / max(1, len(c_tokens)) if c_tokens else 0.0
    score += 0.6 * token_overlap
    r = SequenceMatcher(None, text_norm, cand_norm).ratio()
    score += 0.5 * r
    return score

def search_vendor_images_for_text(vendor: str, detected_text: str, top_k: int = 2):
    if not vendor or not detected_text:
        return []
    vendor_key = None
    vnorm = _normalize_vendor_key(vendor)
    # direct normalized lookup
    if vnorm in VENDOR_IMAGES_NORM:
        vendor_key = VENDOR_IMAGES_NORM[vnorm]
    else:
        # fallback: find any key whose normalized form contains the vendor normalized text
        for nk, orig in VENDOR_IMAGES_NORM.items():
            if vnorm and vnorm in nk:
                vendor_key = orig; break
        if vendor_key is None:
            for nk, orig in VENDOR_IMAGES_NORM.items():
                if nk and nk in vnorm:
                    vendor_key = orig; break
    if not vendor_key:
        return []
    candidates = VENDOR_IMAGES.get(vendor_key, [])
    if not candidates:
        return []
    text_norm = _normalize_basename(detected_text)
    scored = []
    for c in candidates:
        cand_norm = _normalize_basename(c)
        sc = _score_candidate(text_norm, cand_norm)
        scored.append((c, sc))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{'image_basename': name, 'score': round(sc, 4)} for name, sc in scored[:top_k]]

# Generic fallbacks to catch common formats
GENERIC_MODEL_PATTERNS = [
    r"\b[A-Z]{2,6}-\s*\d{2,4}[A-Z0-9\-+]*\b",   # DGS-1210, TL-SG2210, WS-C2960-24TT
    r"\b[A-Z]{2,6}\d{2,4}[A-Z0-9\-+]*\b",       # GS108, C2960X
    r"\b\d{3,4}[A-Z]{0,3}-\d{2}[A-Z+\-]{0,3}\b" # 2960-24TT
]

def _compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

COMPILED_MODEL_PATTERNS = {v: _compile_patterns(ps) for v, ps in MODEL_PATTERNS.items()}
COMPILED_GENERIC_PATTERNS = _compile_patterns(GENERIC_MODEL_PATTERNS)

def extract_model_matches(ocr_text: str, matched_vendor: str = None) -> list:
    """
    Return list of plausible model substrings matched from OCR text.
    Vendor-specific patterns are tried first (if vendor known), then generic patterns.
    """
    if not ocr_text:
        return []
    text = re.sub(r"[\n\r]+", " ", ocr_text)
    text = re.sub(r"\s{2,}", " ", text)

    # Collect matches with scores so we can rank the best candidate first.
    # Vendor-specific matches receive a higher base score so they are preferred.
    found_scores = {}  # match_str -> score

    # Vendor-specific patterns first (higher weight)
    if matched_vendor and matched_vendor in COMPILED_MODEL_PATTERNS:
        for rx in COMPILED_MODEL_PATTERNS[matched_vendor]:
            for m in rx.finditer(text):
                s = m.group(0).replace(" ", "")
                score = 100 + len(s)
                found_scores[s] = max(found_scores.get(s, 0), score)

    # Generic patterns next (lower weight)
    for rx in COMPILED_GENERIC_PATTERNS:
        for m in rx.finditer(text):
            s = m.group(0).replace(" ", "")
            score = 10 + len(s)
            found_scores[s] = max(found_scores.get(s, 0), score)

    # Simple plausibility filter
    def is_plausible(s: str) -> bool:
        s = s.strip()
        if len(s) < 4:
            return False
        bad = {"catalyst", "switch", "smart", "gigabit", "cloud", "series"}
        if s.lower() in bad:
            return False
        return bool(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s))

    # Keep only plausible matches
    plausible = {s: sc for s, sc in found_scores.items() if is_plausible(s)}

# ------------------------------ IMAGE PROCESS ------------------------
def process_image(img_path):
    """Process a single image and return the data dict with detailed debug fields."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Keep raw OCR result
    ocr_full = ocr_image(img)
    ocr_used = ocr_full
    ocr_left = ""
    ocr_right = ""

    # Quick cleaned text length
    text_content = ocr_used.replace(" \n ", "").replace("[No text detected]", "")
    try:
        need_crop_retry = (ocr_used.strip() == "[No text detected]") or (len(text_content) < 4)
    except Exception:
        need_crop_retry = True

    # Try left/right/center crops if needed (wider than before) and small rotations
    if need_crop_retry:
        h, w = img.shape[:2]
        crop_pct = 0.30
        crop_w = max(1, int(w * crop_pct))
        left_img = img[:, :crop_w]
        right_img = img[:, max(0, w - crop_w):]
        # center vertical strip
        cx1 = max(0, w//2 - crop_w//2)
        cx2 = min(w, cx1 + crop_w)
        center_img = img[:, cx1:cx2]
        ocr_left = ocr_image(left_img)
        ocr_right = ocr_image(right_img)
        ocr_center = ocr_image(center_img)
        pieces = []
        if ocr_full and ocr_full.strip() != "[No text detected]":
            pieces.append(ocr_full)
        if ocr_left and ocr_left.strip() != "[No text detected]":
            pieces.append(ocr_left)
        if ocr_right and ocr_right.strip() != "[No text detected]":
            pieces.append(ocr_right)
        if ocr_center and ocr_center.strip() != "[No text detected]":
            pieces.append(ocr_center)
        # If nothing useful, try small rotations of the crops
        if not pieces:
            def rotate(img_in, ang):
                (h2, w2) = img_in.shape[:2]
                M = cv2.getRotationMatrix2D((w2/2, h2/2), ang, 1.0)
                return cv2.warpAffine(img_in, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            angles = [-7, 7]
            for crop_img in (left_img, center_img, right_img):
                for a in angles:
                    try:
                        rot = rotate(crop_img, a)
                        rt = ocr_image(rot)
                        if rt and rt.strip() != "[No text detected]":
                            pieces.append(rt)
                    except Exception:
                        pass
        if pieces:
            # merge conservatively
            ocr_used = " \n ".join(pieces)

    # Classification (vendor)
    text_content = ocr_used.replace(" \n ", "").replace("[No text detected]", "")
    if len(text_content) < 4:
        possible_ocr_vendors = []
        matched_vendor = None
    else:
        classification = classify_detected_text(ocr_used)
        matched_vendor = classification.get('matched_vendor')
        possible_ocr_vendors = [v for v, _ in classification.get('candidates', [])]

    # Model extraction using the text we decided to use
    model_matches = extract_model_matches(ocr_used, matched_vendor=matched_vendor)
    best_model = model_matches[0] if model_matches else None
    detected_text_for_model = ", ".join(model_matches) if model_matches else "—"

    # Search vendor images (if available)
    model_candidates = []
    search_text = detected_text_for_model if detected_text_for_model and detected_text_for_model != "—" else (best_model or "")
    vendor_images_key = None
    vendor_images_count = 0
    if matched_vendor and search_text and VENDOR_IMAGES:
        model_candidates = search_vendor_images_for_text(matched_vendor, search_text, top_k=2)
        # find vendor_images_key/count for debug
        for k in VENDOR_IMAGES.keys():
            if k.lower() == matched_vendor.lower() or matched_vendor.lower() in k.lower():
                vendor_images_key = k
                vendor_images_count = len(VENDOR_IMAGES.get(k, []))
                break
        if model_candidates:
            best_model = model_candidates[0]['image_basename']

    return {
        'img_path': img_path,
        'ocr_full_text': ocr_full,
        'ocr_left_text': ocr_left,
        'ocr_right_text': ocr_right,
        'ocr_used_text': ocr_used,
        'possible_ocr_vendors': possible_ocr_vendors,
        'matched_vendor': matched_vendor,
        'detected_text_for_model': detected_text_for_model,
        'model_matches': model_matches,
        'model': best_model,
        'model_candidates': model_candidates,
        'vendor_images_key': vendor_images_key,
        'vendor_images_count': vendor_images_count
    }


# ------------------------------- MAIN --------------------------------
def main():
    img_paths = list(sorted(IMAGES_FOLDER.glob("*.*")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image, img_paths))

    # Build minimal HTML table with only Image and Predicted Vendor

    html = """
<html>
<head>
<style>
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #444; padding: 8px; vertical-align: top; }
th { background: #f0f0f0; }
img { max-width: 280px; }
</style>
</head>
<body>
<h2>OCR Predicted Vendor</h2>
<table>
<tr><th>Image</th><th>Predicted Vendor</th></tr>
"""

    for result in results:
        if not result:
            continue
        html += "<tr>\n"
        html += f"<td><img src=\"{escape(str(result['img_path']))}\"></td>\n"
        html += f"<td>{escape(str(result.get('matched_vendor') or ''))}</td>\n"
        html += "</tr>\n"

    html += "</table></body></html>"
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

def detect_vendor(image_path):
    result = process_image(Path(image_path))
    if not result:
        return None
    return result