import os, cv2, base64, re, hashlib, pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import easyocr
from fuzzywuzzy import fuzz
import imagehash
import argparse
import zipfile
import tempfile
from skimage.metrics import structural_similarity as ssim

# Fix SSL certificate verification error for EasyOCR model downloads on Windows
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request._create_default_https_context = ssl._create_unverified_context

# ====================== CONFIG ======================

RACK_TYPE_MODEL_PATH = r"Trained_Models\best 2.keras"  # updated
default_model = Path(r"F:\\backend_functionality\\Racktrack0912 (2)\\Racktrack0912\\Trained_Models\\best.keras")
RACK_TYPE_MODEL_PATH = str(default_model) if default_model.exists() else r"Trained_Models\best.keras"
RACK_UNIT_MODEL    = r"Trained_Models\\rack_unit.pt"
SWITCH_PATCH_MODEL = r"Trained_Models\\switch_patch.pt"

# Default (local) paths — may be overridden by job-style layout under `jobs_output/`
INPUT_FOLDER  = Path(r"jobs_output\job_1767333358486_1cb7dfb8\input")
OUTPUT_FOLDER = Path(r"Refers_images\\annotated")
HTML_PATH     = Path(r"Results\\rack_report.html")
PKL_FOLDER    = Path(r"Trained_Models")
CACHE_FILE    = PKL_FOLDER / "ref_features_cache.pkl"

# If a cache file exists in other known locations (workspace Trained_Models
# or the absolute path you provided), prefer that and update `PKL_FOLDER`.
possible_cache_locations = [
    PKL_FOLDER / "ref_features_cache.pkl",
    Path.cwd() / "Trained_Models" / "ref_features_cache.pkl",
    Path(r"F:\\backend_functionality\\Racktrack0912 (2)\\Racktrack0912\\Trained_Models\\ref_features_cache.pkl")
]
for p in possible_cache_locations:
    if p.exists():
        CACHE_FILE = p
        PKL_FOLDER = p.parent
        break

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
RU_CONF  = 0.4
SWITCH_CONF = 0.4

# If a job-style input exists (jobs_output/job_xxx/input), prefer that and
# place outputs under that job's output folder. This matches the example
# path you provided (jobs_output\job_...\input\...jpg).
parser = argparse.ArgumentParser(add_help=False)
# Accept either --job or --job-id for compatibility; also allow explicit input/output paths
parser.add_argument("--job-id", help="Job folder name or id (optional)")
parser.add_argument("--job", help="Job folder name, id, or absolute job path (optional)")
parser.add_argument("--input", help="Explicit input folder containing images (optional)")
parser.add_argument("--output", help="Explicit output folder or html file (optional)")
args, _ = parser.parse_known_args()
JOB_ARG = args.job or args.job_id or os.environ.get("JOB_ID") or os.environ.get("JOBID")

JOBS_ROOT = Path("jobs_output")
# If explicit input/output paths provided, prefer them
if getattr(args, "input", None):
    p = Path(args.input)
    if p.exists():
        INPUT_FOLDER = p
if getattr(args, "output", None):
    outp = Path(args.output)
    # if output is a file path (ends with .html), write HTML there; else treat as folder
    if outp.suffix.lower() in {".html", ".htm"}:
        HTML_PATH = outp
        HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FOLDER = HTML_PATH.parent / "Refers_images" / "annotated"
    else:
        OUTPUT_FOLDER = outp / "Refers_images" / "annotated"
        HTML_PATH = outp / "Results" / "rack_report.html"
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
if JOBS_ROOT.exists():
    # If user supplied a job arg, try resolve. Accept absolute job folder paths too.
    if JOB_ARG:
        candidates = []
        arg_path = Path(JOB_ARG)
        # If provided value is an existing path, treat it as the job folder directly
        if arg_path.exists() and arg_path.is_dir():
            candidates = [arg_path]
        else:
            # exact match under jobs_output
            jd_exact = JOBS_ROOT / JOB_ARG
            if jd_exact.exists():
                candidates = [jd_exact]
            else:
                # prefixed form job_<id>
                jd_pref = JOBS_ROOT / f"job_{JOB_ARG}"
                if jd_pref.exists():
                    candidates = [jd_pref]
                else:
                    # any directory starting with JOB_ARG or job_JOBARG
                    candidates = [p for p in JOBS_ROOT.iterdir() if p.is_dir() and (p.name.startswith(JOB_ARG) or p.name.startswith(f"job_{JOB_ARG}"))]
        if candidates:
            jd = sorted(candidates, reverse=True)[0]
            # prefer explicit 'input' folder, but if not present look in common
            # locations (output/rack, output) or recursively find any image
            def find_images_in_job_dir(job_dir: Path):
                # check common subfolders first
                locs = [job_dir / 'input', job_dir / 'images', job_dir / 'input' / 'images', job_dir / 'output' / 'rack', job_dir / 'output']
                for p in locs:
                    if p.exists() and p.is_dir():
                        imgs = [q for q in p.glob('*') if q.suffix.lower() in IMG_EXTS]
                        if imgs:
                            return p, imgs
                # recursive search for any image under job_dir
                imgs = list(job_dir.rglob('*'))
                imgs = [q for q in imgs if q.is_file() and q.suffix.lower() in IMG_EXTS]
                if imgs:
                    return imgs[0].parent, imgs
                return None, []

            cand = jd / "input"
            imgs = []
            if cand.exists() and cand.is_dir():
                imgs = [p for p in cand.glob("*") if p.suffix.lower() in IMG_EXTS]
            if not imgs:
                found_parent, imgs = find_images_in_job_dir(jd)
                if found_parent:
                    INPUT_FOLDER = found_parent
                else:
                    INPUT_FOLDER = cand  # keep default even if empty
            else:
                INPUT_FOLDER = cand

            if imgs:
                OUTPUT_FOLDER = jd / "output" / "Refers_images" / "annotated"
                HTML_PATH = jd / "output" / "Results" / "rack_report.html"
                # Prefer an existing global cache if present to avoid rebuilding per-job
                candidate_pkl = jd / "output" / "Trained_Models"
                candidate_cache = candidate_pkl / "ref_features_cache.pkl"
                found_global = next((p for p in possible_cache_locations if p.exists()), None)
                if found_global:
                    CACHE_FILE = found_global
                    PKL_FOLDER = found_global.parent
                else:
                    PKL_FOLDER = candidate_pkl
                    CACHE_FILE = candidate_cache
                print(f"Resolved job folder: {jd}")
                print(f"Using input folder: {INPUT_FOLDER} (found {len(imgs)} image(s))")
            else:
                # no images under job; still set outputs for consistency
                OUTPUT_FOLDER = jd / "output" / "Refers_images" / "annotated"
                HTML_PATH = jd / "output" / "Results" / "rack_report.html"
                candidate_pkl = jd / "output" / "Trained_Models"
                candidate_cache = candidate_pkl / "ref_features_cache.pkl"
                found_global = next((p for p in possible_cache_locations if p.exists()), None)
                if found_global:
                    CACHE_FILE = found_global
                    PKL_FOLDER = found_global.parent
                else:
                    PKL_FOLDER = candidate_pkl
                    CACHE_FILE = candidate_cache
                print(f"Resolved job folder: {jd} — no images found in common locations")
    else:
        # Try to infer job folder from current working directory first (useful for JobWorker)
        cwd = Path.cwd()
        inferred = None
        for anc in [cwd] + list(cwd.parents):
            if anc.name.startswith("job_") and anc.is_dir():
                inferred = anc
                break
        if inferred:
            cand = inferred / "input"
            imgs = []
            if cand.exists() and cand.is_dir():
                imgs = [p for p in cand.glob("*") if p.suffix.lower() in IMG_EXTS]
            if not imgs:
                # try other common locations under the inferred job folder
                found_parent, imgs = (None, [])
                def _search_job(jd):
                    locs = [jd / 'input', jd / 'images', jd / 'input' / 'images', jd / 'output' / 'rack', jd / 'output']
                    for p in locs:
                        if p.exists() and p.is_dir():
                            l = [q for q in p.glob('*') if q.suffix.lower() in IMG_EXTS]
                            if l:
                                return p, l
                    all_imgs = list(inferred.rglob('*'))
                    all_imgs = [q for q in all_imgs if q.is_file() and q.suffix.lower() in IMG_EXTS]
                    return (all_imgs[0].parent, all_imgs) if all_imgs else (None, [])
                found_parent, imgs = _search_job(inferred)
                if found_parent:
                    INPUT_FOLDER = found_parent
            else:
                INPUT_FOLDER = cand
            if imgs:
                OUTPUT_FOLDER = inferred / "output" / "Refers_images" / "annotated"
                HTML_PATH = inferred / "output" / "Results" / "rack_report.html"
                PKL_FOLDER = inferred / "output" / "Trained_Models"
                CACHE_FILE = PKL_FOLDER / "ref_features_cache.pkl"
                print(f"Inferred job folder from cwd: {inferred} (found {len(imgs)} image(s))")
        else:
            # fallback: pick the latest job_* directory containing an input folder with images
            job_dirs = sorted([p for p in JOBS_ROOT.iterdir() if p.is_dir() and p.name.startswith("job_")], reverse=True)
            for jd in job_dirs:
                cand = jd / "input"
                if cand.exists():
                    imgs = [p for p in cand.glob("*") if p.suffix.lower() in IMG_EXTS]
                    if imgs:
                        INPUT_FOLDER = cand
                        OUTPUT_FOLDER = jd / "output" / "Refers_images" / "annotated"
                        HTML_PATH = jd / "output" / "Results" / "rack_report.html"
                        PKL_FOLDER = jd / "output" / "Trained_Models"
                        CACHE_FILE = PKL_FOLDER / "ref_features_cache.pkl"
                        print(f"Auto-selected latest job folder: {jd}")
                        break

# ensure folders exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
PKL_FOLDER.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["enclosed","openframe","wall mounted"]

COLOR_SWITCH     = (0,200,0)
COLOR_PATCHPANEL = (0,140,255)
COLOR_EMPTY      = (160,160,160)

REF_DIRS = {
    "enclosed":    Path(r"C:\Users\AasrithaSravaniBhami\Downloads\reference_images\reference_images\Enclosed_Rack"),
    "openframe":   Path(r"C:\Users\AasrithaSravaniBhami\Downloads\reference_images\reference_images\OpenFrame_Rack"),
    "wall mounted": Path(r"C:\Users\AasrithaSravaniBhami\Downloads\reference_images\reference_images\WallMount_Rack"),
}

# ====================== HELPERS ======================

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

def ordinal(n):
    return f"{n}{'st' if n%10==1 and n%100!=11 else 'nd' if n%10==2 and n%100!=12 else 'rd' if n%10==3 and n%100!=13 else 'th'}"

def read_to_data_uri(path):
    img = cv2.imread(str(path))
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def color_for(label):
    if label == "switch": return COLOR_SWITCH
    if label == "patch panel": return COLOR_PATCHPANEL
    return COLOR_EMPTY

# ====================== OCR ======================

VENDORS = [
 "MTS Infonet","SRG International","Netrack1","LYNN Broadband","IOGear","Alpine Racks","Amco Enclosure",
 "ATEn International","B&R Enclosure","Bch India","Belden","Black Box","Cannon Technologies","Canovate",
 "Centerpeer Corporation","Chatsworth Products","Conteg","CPI (Chatsworth Products)","Data Rack Technologies",
 "Deckro","Didgitage Infocom Private Limited","EA-HWA","Eaton Corporation","Electron Metal","Elixir Network",
 "Finen","Flexsol","Great Lakes Data Rack & Cabinets","Hammond","Hanut India","Hardy","Hewlett Packard Enterprise (HPE)",
 "ITRack","Jyotitech Industries","Kendell Howard","Legrand","Martin International Enclosures","Metworks System",
 "Middle Atlantic Products","Modular Rack Systems","Netrack","Netshell","nVent Hoffman","nVent Schroff","Panduit",
 "R&M","Rackmount Solutions","Rackom Systems","Rack Solutions","Rxube Infotech","Rittal","RXL","Schneider",
 "Startech","Suntec","Sysrack","Trend Tech Industries","Vertiv","WQIndia","NavePoint","Pulsar","INTELLINET",
 "TECMOJO","YIPU Communication","HTYZ","TECHLY","LINKRacks","GIGANET","Lanberg","Rackmatic","BlueStrome",
 "CERTECH","Gcabling","FS","Aeons Technologies","Catvscope","EXTELL","Skeletek","ECHOGEAR","Rackpath",
 "Prorack","Wavenet","Cmple","Linkbasic","Toten","ARMADI","Lande","19InchRacks","Link-X","ITK","MASAR","Triton Systems","Samson"
]

_OCR_READER = easyocr.Reader(['en'], gpu=False)
_OCR_CACHE = {}

def clean_text(text): 
    return re.sub(r'[^a-zA-Z0-9\\s]', '', text).lower().strip()

def best_fuzzy_match(text, vendors):
    text = clean_text(text)
    best_vendor, best_score = None, 0
    for vendor in vendors:
        v_clean = clean_text(vendor)
        score = max(fuzz.partial_ratio(text, v_clean),
                    fuzz.token_set_ratio(text, v_clean),
                    fuzz.token_sort_ratio(text, v_clean))
        if score > best_score:
            best_vendor, best_score = vendor, score
    return best_vendor, best_score

def _md5(path: Path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _prep_for_ocr(img):
    if img is None or img.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Avoid upscaling blindly (can cause huge arrays and OOM for large inputs).
    # Choose a reasonable maximum side length for OCR preprocessing.
    MAX_OCR_SIDE = 1024
    h, w = gray.shape[:2]
    # default upscale factor used previously; allow downscale if image too large
    target_scale = 1.3
    # compute scale to ensure max side <= MAX_OCR_SIDE
    if max(h, w) * target_scale > MAX_OCR_SIDE:
        target_scale = MAX_OCR_SIDE / float(max(h, w))
        if target_scale <= 0:
            target_scale = 1.0
    # apply scaling (could be <1 for large images)
    new_w = max(1, int(w * target_scale))
    new_h = max(1, int(h * target_scale))
    try:
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        gray = cv2.resize(gray, (min(new_w, 1024), min(new_h, 1024)), interpolation=cv2.INTER_AREA)
    return gray

def detect_vendor_ocr(image_path: Path, fuzzy_thresh=65):
    key = _md5(image_path)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]
    img = cv2.imread(str(image_path))
    if img is None:
        return {'vendor': None, 'ocr_text': ''}
    h, w = img.shape[:2]
    bands = [
        img[0:int(h * 0.3)],
        img[int(h * 0.35):int(h * 0.65)],
        img[int(h * 0.7):h],
        img
    ]
    best = (None, 0)
    all_texts = []
    for band in bands:
        g = _prep_for_ocr(band)
        try:
            texts = _OCR_READER.readtext(g, detail=0, paragraph=True)
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            # fallback: try smaller scale
            try:
                small = cv2.resize(g, (max(1, g.shape[1]//2), max(1, g.shape[0]//2)), interpolation=cv2.INTER_AREA)
                texts = _OCR_READER.readtext(small, detail=0, paragraph=True)
            except Exception:
                texts = []
        except Exception:
            # any other OCR failure should not crash the pipeline; skip this band
            texts = []
        if not texts:
            continue
        txt = " ".join(texts)
        cleaned = clean_text(txt)
        all_texts.append(cleaned)
        for vendor in VENDORS:
            if clean_text(vendor) in cleaned:
                result = {'vendor': vendor, 'ocr_text': " | ".join(all_texts)}
                _OCR_CACHE[key] = result
                return result
        cand, score = best_fuzzy_match(cleaned, VENDORS)
        if score > best[1]:
            best = (cand, score)
    vendor = best[0] if best[1] >= fuzzy_thresh else None
    result = {'vendor': vendor, 'ocr_text': " | ".join(all_texts)}
    _OCR_CACHE[key] = result
    return result

# ====================== UPDATED RACK TYPE MODEL ======================

def load_rack_type_model(path):
    p = Path(path)
    if not p.exists():
        alt = p.with_suffix(".keras")
        if alt.exists():
            p = alt
        else:
            print(f"⚠️ Rack type model not found: {p} — model disabled, using heuristic")
            return None
    # Try several safe loading strategies to handle models saved with different configs
    try:
        return tf.keras.models.load_model(str(p), compile=False, safe_mode=True)
    except Exception as e1:
        try:
            return tf.keras.models.load_model(str(p), compile=False)
        except Exception as e2:
            print(f"⚠️ Failed to load rack type model ({p}): {e2}\nAttempting weights-by-name fallback from .keras archive...")
            # If the model is a Keras archive (.keras) containing model.weights.h5,
            # try to construct a compatible model and load weights by name.
            try:
                if zipfile.is_zipfile(str(p)):
                    with zipfile.ZipFile(str(p), 'r') as z:
                        members = z.namelist()
                        if 'model.weights.h5' in members:
                            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                                tmp_path = tmp.name
                                tmp.write(z.read('model.weights.h5'))
                            # Attempt to build a model compatible with common setups.
                            for size in [(300, 300), (301, 301)]:
                                try:
                                    inp = tf.keras.layers.Input(shape=(size[0], size[1], 3))
                                    base = tf.keras.applications.EfficientNetB3(
                                        weights=None,
                                        include_top=False,
                                        pooling='avg',
                                        input_tensor=inp
                                    )
                                    out = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(base.output)
                                    model = tf.keras.models.Model(inputs=inp, outputs=out)
                                    model.load_weights(tmp_path, by_name=True)
                                    print(f"✅ Successfully loaded weights by_name into EfficientNetB3-based model (input {size[0]}x{size[1]}x3)")
                                    return model
                                except Exception as e3:
                                    # try next size
                                    continue
                print("⚠️ Weights-by-name fallback failed — model will be disabled and heuristic used.")
            except Exception as e_fallback:
                print(f"⚠️ Exception during weights-by-name fallback: {e_fallback}")
            return None

rack_type_model = load_rack_type_model(RACK_TYPE_MODEL_PATH)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def infer_rack_type(img):
    """Accurate rack type classification using EfficientNetB3 model."""
    try:
        # quick geometric heuristic for wall-mounted racks:
        # - if image width > height, or image is approximately square,
        #   classify immediately as 'wall mounted'
        h, w = img.shape[:2]
        ASPECT_SQUARE_THRESH = 0.10  # relative difference threshold (10%)
        is_square_like = abs(w - h) / float(max(w, h)) <= ASPECT_SQUARE_THRESH
        if w > h or is_square_like:
            print(f"   🔹 Heuristic: classified as 'wall mounted' (w={w}, h={h})")
            return "wall mounted"

        # Otherwise use the trained model if available. If the model
        # predicts 'wall mounted' despite the geometry, force a choice
        # between the remaining two classes ('enclosed' or 'openframe').
        if rack_type_model is None:
            print("⚠️ Rack type model not available — returning 'unknown'")
            return "unknown"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb).convert("RGB")
        arr = np.array(im_pil.resize((300, 300), resample=Image.BILINEAR), dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        pred = rack_type_model.predict(arr, verbose=0)
        pred = np.array(pred)
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = pred[:, 0, :]
        if pred.ndim != 2:
            raise ValueError(f"Unexpected prediction shape {pred.shape}")
        probs = pred
        if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3):
            probs = softmax(probs)
        probs = probs[0]

        # If model prefers 'wall mounted' despite geometric test, pick
        # the higher of 'enclosed' / 'openframe' instead.
        top_idx = int(np.argmax(probs))
        top_name = CLASS_NAMES[top_idx]
        if top_name == "wall mounted":
            # choose between the other two classes using their probabilities
            try:
                idx_enclosed = CLASS_NAMES.index("enclosed")
                idx_open = CLASS_NAMES.index("openframe")
                pick_idx = int(np.argmax([probs[idx_enclosed], probs[idx_open]]))
                top_name = ["enclosed", "openframe"][pick_idx]
                conf = float(probs[idx_enclosed] if pick_idx == 0 else probs[idx_open])
            except Exception:
                # fallback: keep model choice if anything goes wrong
                conf = float(probs[top_idx])
        else:
            conf = float(probs[top_idx])

        print(f"   🔹 Rack Type Prediction: {top_name} (Confidence: {conf:.3f})")
        return top_name
    except Exception as e:
        print(f"⚠️ Rack type inference failed: {e}")
        return "unknown"

# ====================== YOLO MODELS ======================

ru_model = YOLO(RACK_UNIT_MODEL)
sp_model = YOLO(SWITCH_PATCH_MODEL)
SP_NAMES = sp_model.names if hasattr(sp_model, "names") else {0:"patch_panel",1:"switch"}

# ====================== FEATURE EXTRACTION ======================

try:
    feat_extractor = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(300, 300, 3)
    )
except Exception as e:
    print(f"⚠️ Failed to load EfficientNetB3 with imagenet weights: {e}\nFalling back to EfficientNetB3 without pretrained weights.")
    try:
        feat_extractor = tf.keras.applications.EfficientNetB3(
            weights=None,
            include_top=False,
            pooling="avg",
            input_shape=(300, 300, 3)
        )
    except Exception as e2:
        print(f"⚠️ Failed to construct EfficientNetB3 fallback: {e2}\nFeature extraction will be disabled.")
        feat_extractor = None

def extract_feature(img_path: Path):
    with Image.open(img_path).convert("RGB") as im:
        arr = np.array(im.resize((300, 300)), dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0)
        feat = feat_extractor.predict(arr, verbose=0)[0]
        feat = feat / np.linalg.norm(feat)
    return feat

def load_reference_features():
    needs_rebuild = False
    ref_data = {}
    if Path(CACHE_FILE).exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                ref_data = pickle.load(f)
            any_valid = any(Path(item["path"]).exists() for group in ref_data.values() for item in group)
            if not any_valid:
                print("⚠️ Cached paths invalid. Rebuilding reference feature cache ...")
                needs_rebuild = True
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}. Rebuilding ...")
            needs_rebuild = True
    else:
        needs_rebuild = True
    if not needs_rebuild:
        print("✅ Loaded reference cache successfully.")
        return ref_data
    print("🔁 Rebuilding reference feature cache ...")
    ref_data = {}
    for cname, folder in REF_DIRS.items():
        if not folder.exists():
            print(f"⚠️ Folder not found: {folder}")
            ref_data[cname] = []
            continue
        imgs = [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS]
        feats_list = []
        for p in imgs:
            try:
                feat = extract_feature(p)
                feats_list.append({"feat": feat, "path": str(p)})
            except Exception as e:
                print(f"   ⚠️ Failed to process {p}: {e}")
                continue
        ref_data[cname] = feats_list
        print(f"✅ {cname}: cached {len(feats_list)} images.")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(ref_data, f)
    print("✅ Reference feature cache rebuilt successfully.")
    return ref_data

# ====================== HYBRID MATCHING ======================

def edge_similarity(img1, img2):
    def get_edges(im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.resize(edges, (300, 300))
    e1, e2 = get_edges(img1), get_edges(img2)
    try:
        return ssim(e1, e2)
    except Exception:
        return 0.0

def parse_vendor_from_filename(path_str):
    stem = Path(path_str).stem
    tokens = re.split(r"[ _\-|]+", stem)
    best, best_score = None, 0
    for t in tokens[:4]:
        cand, sc = best_fuzzy_match(t, VENDORS)
        if sc > best_score:
            best, best_score = cand, sc
    return best if best else tokens[0] if tokens else stem

def match_rack_via_hybrid(img_path: Path, rack_type: str, ref_data, match_stats=None, score_thresh=0.25):
    if match_stats is None:
        match_stats = {}
    folder_map = {
        "enclosed": "enclosed",
        "openframe": "openframe",
        "wallmounted": "wall mounted",
        "wall mounted": "wall mounted",
        "wall-mounted": "wall mounted"
    }
    rack_type_norm = folder_map.get(rack_type.lower().strip(), rack_type.lower().strip())
    refs = ref_data.get(rack_type_norm, [])
    print(f"\n🧩 Rack type: '{rack_type_norm}' | Found {len(refs)} reference entries in cache")
    if not refs:
        return None
    img_in = cv2.imread(str(img_path))
    if img_in is None:
        return None
    edge_scores = []
    for r in refs:
        ref_path = r["path"]
        img_ref = cv2.imread(ref_path)
        if img_ref is None:
            continue
        es = edge_similarity(img_in, img_ref)
        edge_scores.append((es, r))
    if not edge_scores:
        print("⚠️ No edge scores computed.")
        return None
    edge_scores.sort(key=lambda x: x[0], reverse=True)
    top_refs = [r for _, r in edge_scores[:5]]
    print("✅ Computed edge scores. Re-ranking top 5 with EfficientNet ...")
    feat_in = extract_feature(img_path)
    best_match, best_score = None, -1.0
    for r in top_refs:
        feat_ref = r["feat"]
        feat_sim = float(np.dot(feat_in, feat_ref))
        edge_sim = [es for es, rr in edge_scores if rr["path"] == r["path"]]
        edge_sim = float(edge_sim[0]) if edge_sim else 0.0
        edge_norm = max(0.0, min(edge_sim, 1.0))
        feat_norm = (feat_sim + 1.0) / 2.0
        vendor_name = parse_vendor_from_filename(r["path"])
        repeats = match_stats.get(vendor_name, 0)
        penalty = 0.05 * repeats
        final_score = 0.55 * edge_norm + 0.45 * feat_norm - penalty
        print(f"   🔸 {Path(r['path']).stem} → Edge:{edge_sim:.3f} Feat:{feat_sim:.3f} Final:{final_score:.3f} Vendor:{vendor_name}")
        if final_score > best_score:
            best_score = final_score
            best_match = {"path": r["path"], "feat": feat_ref, "score": final_score, "vendor": vendor_name}
    if best_match and best_match["score"] >= score_thresh:
        print(f"✅ Best match: {Path(best_match['path']).stem} (score={best_match['score']:.3f})")
        match_stats[best_match["vendor"]] = match_stats.get(best_match["vendor"], 0) + 1
        return best_match
    else:
        print(f"❌ No confident match found (best={best_score:.3f}, threshold={score_thresh})")
        return None

# ====================== DETECTION ======================

def classify_units_with_full_detection(img, ru_boxes):
    results = sp_model(img, conf=SWITCH_CONF, verbose=False)[0]
    detections = []
    for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                 results.boxes.cls.cpu().numpy().astype(int),
                                 results.boxes.conf.cpu().numpy()):
        raw = SP_NAMES[int(cls_id)].lower()
        label = "patch panel" if "patch" in raw else "switch" if "switch" in raw else None
        if label:
            detections.append((label, conf, box))
    ru_labels = []
    for i, ru in enumerate(ru_boxes, 1):
        best_label, best_conf = "empty", 0.0
        for label, conf, box in detections:
            if iou(ru, box) > 0.25 and conf > best_conf:
                best_label, best_conf = label, conf
        ru_labels.append((best_label, best_conf))
    return ru_labels

# ====================== PROCESS IMAGE ======================

def process_image(img_path, ref_data, match_stats):
    img = cv2.imread(str(img_path))
    if img is None: return None
    rack_type = infer_rack_type(img)
    ru_results = ru_model.predict(img, conf=RU_CONF, verbose=False)[0]
    ru_boxes = sorted(ru_results.boxes.xyxy.cpu().numpy(), key=lambda b: (b[1]+b[3])/2)
    ru_labels = classify_units_with_full_detection(img, ru_boxes)
    annotated = img.copy()
    summary = []
    for i, ((x1,y1,x2,y2), (label,conf)) in enumerate(zip(ru_boxes, ru_labels),1):
        color = color_for(label)
        cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        cv2.putText(annotated,f"{i}U - {label}",(int(x1),max(15,int(y1)-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        summary.append(f"{ordinal(i)} unit – {label}")
    ann_path = Path(OUTPUT_FOLDER)/(img_path.stem+"_ann.png")
    cv2.imwrite(str(ann_path), annotated)
    vendor_result = detect_vendor_ocr(img_path)
    vendor = vendor_result['vendor']
    ocr_text = vendor_result['ocr_text']
    if vendor:
        vendor_source = "OCR"
    else:
        match = match_rack_via_hybrid(img_path, rack_type, ref_data, match_stats=match_stats, score_thresh=0.25)
        if match:
            vendor = Path(match["path"]).stem
            vendor_source = f"Hybrid Match (Score: {match['score']:.3f})"
        else:
            vendor = "Not detected"
            vendor_source = "None"
    print(f"Vendor for {img_path.name}: {vendor} ({vendor_source})")
    return {
        "filename": img_path.name,
        "image": read_to_data_uri(img_path),
        "annotated": read_to_data_uri(ann_path),
        "rack_type": rack_type,
        "vendor": vendor,
        "ocr_text": ocr_text,
        "summary": summary or ["No rack units detected"]
    }

# ====================== HTML REPORT ======================

def build_html(data):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
    <html><head><style>
    body{{font-family:Arial;padding:20px}}
    table{{width:100%;border-collapse:collapse}}
    th,td{{border:1px solid #aaa;padding:10px;text-align:center}}
    img{{max-height:220px}}
    </style></head><body>
    <h2>Rack Analysis Report</h2>
    <p>Generated: {ts}</p>
    <table>
    <tr><th>Rack</th><th>Rack Type & Vendor</th><th>Annotated</th><th>Summary</th><th>Detected OCR</th></tr>"""
    for d in data:
        html += f"""
        <tr>
        <td><img src="{d['image']}"></td>
        <td><b>{d['rack_type'].upper()}</b><br>{d['vendor']}</td>
        <td><img src="{d['annotated']}"></td>
        <td><ul>{"".join(f"<li>{s}</li>" for s in d["summary"])}</ul></td>
        <td>{d['ocr_text']}</td>
        </tr>"""
    html += "</table></body></html>"
    return html

# ====================== MAIN ======================

def main():
    ref_data = load_reference_features()
    images = [p for p in Path(INPUT_FOLDER).glob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        print("No images found.")
        return
    sections = []
    match_stats = {}
    for img in images:
        print(f"\n🖼️ Processing {img.name} ...")
        result = process_image(img, ref_data, match_stats)
        if result:
            sections.append(result)
    html = build_html(sections)
    with open(HTML_PATH,"w",encoding="utf-8") as f:
        f.write(html)
    print("\n✅ DONE")
    print("✅ HTML:", HTML_PATH)

if __name__ == "__main__":
    main()
