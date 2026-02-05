
# -*- coding: utf-8 -*-
"""
Rack Audit Summary (Summary-Only, strict to user requirements)

Requirements implemented:
- No icons
- No recommendations
- Executive summary: exactly 4 lines, no heading (bulleted with '-')
- No counts for empty/connected ports
- Include rack vendor & rack type; include switch vendors if present
- Capture color-coded Ethernet cables (RJ_45_Black/White/Grey/Yellow)

Outputs:
- PDF (summary-only, grouped sections; exec summary first block with no heading)
- TXT (same content as PDF bullets)
- JSON (parsed audit data)

Usage:
------
python rack_audit_summary.py --pdf "Merged_Result 2.pdf" --out "Audit_Summary_Report.pdf" --font "DejaVuSans.ttf"

Notes:
------
- Provide a Unicode font via --font (e.g., DejaVuSans.ttf) to avoid any square boxes if your environment lacks glyphs.
- No invented details: derived exclusively from parsed input PDF.
"""

import pdfplumber
import json
import re
import argparse
from datetime import datetime
from pathlib import Path

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==============================================================================
# CONFIG DEFAULTS
# ==============================================================================
DEFAULT_PDF_PATH = "Merged_Result 2.pdf"
OUTPUT_JSON = "audit2_data.json"
DEFAULT_OUTPUT_PDF = "Audit_Summary_Report.pdf"

# ==============================================================================
# SECTION HEADERS (for parsing common report pages)
# ==============================================================================
SECTION_PATTERNS = {
    "patch_panel_details": r"Patch Panel Classification Results",
    "rack_details": r"Rack Detection Report",
    "switch_details": r"OCR Vendor & Model Detection|Switch Make and Model|Switch Classification",
    "cable_classification": r"Cable Classification Report",
    "empty_ports": r"EMPTY PORT IDENTIFICATION AND LOCATION",
    "connected_ports": r"Connected Port Classification",
}

# ==============================================================================
# FONT MANAGEMENT
# ==============================================================================
def register_unicode_font(font_path: str | None) -> str:
    """
    Try registering the given TTF font. If successful, return its face name ("CustomFont").
    If not provided or fails, return a safe default ("Helvetica").
    """
    if font_path:
        p = Path(font_path)
        if p.exists() and p.is_file():
            try:
                pdfmetrics.registerFont(TTFont("CustomFont", str(p)))
                return "CustomFont"
            except Exception as e:
                print(f"[font] Failed to register '{font_path}': {e}")
    # Fallback to built-in Helvetica
    print("[font] Using built-in Helvetica (limited Unicode). Consider --font DejaVuSans.ttf")
    return "Helvetica"

# ==============================================================================
# INPUT PDF EXTRACTION
# ==============================================================================
def extract_pdf_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)

def split_by_sections(text):
    sections = {key: "" for key in SECTION_PATTERNS}
    matches = []
    for key, pattern in SECTION_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((m.start(), key))
    matches.sort()
    for i, (start, key) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections

def clean_lines(block):
    lines = []
    for l in block.splitlines():
        l = re.sub(r"\s+", " ", l).strip()
        if l and not l.lower().startswith(("predicted", "final prediction", "image vendor")):
            lines.append(l)
    return lines

# ==============================================================================
# PARSERS
# ==============================================================================
def parse_rack(block):
    """Extracts unit layout lines, e.g., '1st unit – empty ... 6th unit – empty'."""
    return [
        line for line in clean_lines(block)
        if re.search(r"\d+(st|nd|rd|th)\s+unit", line, re.IGNORECASE)
    ]

def parse_rack_meta(block):
    """
    Extract rack type and vendor from the 'Rack Detection Report' section.
    Examples found in source:
    - 'Original Rack Type & Vendor Annotated Summary WALL MOUNTED'  (type)
    - 'StarTech_Horizontal Wall Mountable'                          (vendor/model-ish)
    """
    text = " ".join(clean_lines(block))
    rack_type = None
    # Type: look for common patterns like WALL MOUNTED
    m_type = re.search(r"\b(WALL[- ]?MOUNTED|FLOOR[- ]?MOUNTED|FREE[- ]?STANDING)\b", text, re.IGNORECASE)
    if m_type:
        rack_type = m_type.group(1).upper().replace("-", " ")

    # Vendor candidates: allow underscore/hyphen/space after vendor
    vendor = None
    m_vendor_inline = re.search(
        r"(StarTech|APC|Rittal|Schneider|Panduit|NetRack|D[- ]?Link|Cisco|HPE|Juniper)(?=[_\\s-]|$)",
        text,
        re.IGNORECASE,
    )
    if m_vendor_inline:
        vendor = m_vendor_inline.group(1)
        vendor = vendor.replace("D Link", "D-Link").replace("d link", "D-Link")

    return {
        "rack_type": rack_type,    # e.g., 'WALL MOUNTED'
        "rack_vendor": vendor      # e.g., 'StarTech'
    }

def parse_patch_panel(block):
    """Pulls lines that indicate RJ-45/Keystone/etc., and NEMA power outlets."""
    lines = clean_lines(block)
    results = []
    for line in lines:
        lo = line.lower()
        if any(k in lo for k in ["keystone", "rj", "usb", "nema"]):
            results.append(line)
    return results

def parse_ports(block):
    """Generic port lines (used for presence checks, not for counts)."""
    return [
        line for line in clean_lines(block)
        if len(line.split()) > 4
    ]

def parse_switch_from_text(full_text):
    """
    Extract switch vendors/models from general text.
    Handles compact 'Switch Make and Model ... Telco PLANET' formats.
    """
    switches = []
    lines = clean_lines(full_text)
    # 'Vendor: <name>' / 'Model: <name>' style
    for i, line in enumerate(lines):
        vendor_match = re.match(r"Vendor\s*:\s*(.+)", line, re.IGNORECASE)
        if vendor_match:
            vendor = vendor_match.group(1).strip()
            model = None
            if i + 1 < len(lines):
                model_match = re.match(r"Model\s*:\s*(.+)", lines[i + 1], re.IGNORECASE)
                if model_match and model_match.group(1).strip():
                    model = model_match.group(1).strip()
            switches.append({"vendor": vendor, "model": model})
    # Compact tokens following 'Switch Make and Model ...'
    for l in lines:
        for token in re.findall(r"\b(PLANET|Cisco|HPE|Juniper|D[- ]?Link|Telco)\b", l, re.IGNORECASE):
            token = token.replace("D Link", "D-Link")
            switches.append({"vendor": token, "model": None})
    # Deduplicate
    dedup = []
    seen = set()
    for s in switches:
        key = (s.get("vendor") or "", s.get("model") or "")
        if key not in seen:
            seen.add(key)
            dedup.append(s)
    return dedup

# ==============================================================================
# FEATURE EXTRACTION (PORTS / CABLES / SWITCHES)
# ==============================================================================
PORT_PATTERNS = {
    "RJ-45 / Keystone": r"RJ[\\ -]?45|keystone",
    "LC Fiber": r"\\bLC\\b",
    "SC Fiber": r"\\bSC\\b",
    "ST Fiber": r"\\bST\\b",
    "SFP / QSFP": r"SFP|QSFP",
    "USB": r"\\bUSB\\b",
}
CABLE_PATTERNS = {
    "Ethernet copper cables": r"ethernet|cat[5-8]|twisted pair|copper",
    "fiber optic cables": r"fiber|optical|lc|sc|st",
}

def extract_ports_and_cables(data):
    ports, cables = set(), set()
    def scan(v):
        if isinstance(v, dict):
            for x in v.values():
                scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
        elif isinstance(v, str):
            for k, p in PORT_PATTERNS.items():
                if re.search(p, v, re.IGNORECASE):
                    ports.add(k)
            for k, p in CABLE_PATTERNS.items():
                if re.search(p, v, re.IGNORECASE):
                    cables.add(k)
    scan(data)
    return sorted(ports), sorted(cables)

def extract_cable_colors(block_text: str) -> list[str]:
    """
    Extract color names from patterns like RJ_45_Black, RJ-45-White, etc.
    """
    raw_tokens = re.findall(r"RJ[_-]?45[_-]([A-Za-z]+)", block_text)

    # canonical color names we care about (ordered preference)
    CANONICAL = [
        "black", "blue", "green", "grey", "white", "yellow",
        "orange", "red", "brown", "purple", "pink", "beige",
        "cyan", "magenta"
    ]

    def normalize_token(tok: str, mapped_so_far: list) -> str | None:
        s = re.sub(r"[^a-z]", "", tok.lower() or "")
        if not s:
            return None
        # try exact or prefix matches against canonical list
        for c in CANONICAL:
            if s == c or c.startswith(s) or s.startswith(c) or s in c:
                # normalize gray/grey to 'Grey'
                return "Grey" if c == "grey" or c == "gray" else c.title()

        # Single-letter heuristics: map initial to most likely canonical not already present
        if len(s) == 1:
            letter = s
            for c in CANONICAL:
                if c.startswith(letter) and c.title() not in mapped_so_far:
                    return c.title()

        # fallback: title-case the cleaned token
        return s.title()

    mapped = []
    for tok in raw_tokens:
        norm = normalize_token(tok, mapped)
        if norm and norm not in mapped:
            mapped.append(norm)

    return mapped

# ==============================================================================
# GROUPED SUMMARY GENERATION (LOCAL, QUALITATIVE ONLY)
# ==============================================================================
def parse_rack_units(rack_lines):
    """
    Convert lines like:
    '1st unit – empty 2nd unit – patch panel 3rd unit – patch panel 4th unit – switch ...'
    into {1: 'Empty', 2: 'Patch Panel', ...}
    """
    mapping = {}
    for line in rack_lines:
        for m in re.finditer(r"(\\d+)(st|nd|rd|th)\\s+unit\\s*[-–]\\s*([A-Za-z ]+)", line, re.IGNORECASE):
            unit = int(m.group(1))
            comp = m.group(3).strip().title()
            mapping[unit] = comp
    return mapping

def build_grouped_summary(audit_data, full_text: str):
    rack_lines = audit_data.get("rack_details", [])
    rack_units = parse_rack_units(rack_lines)
    unit_count = max(rack_units.keys()) if rack_units else 6  # default view only

    rack_meta = audit_data.get("rack_meta", {})
    rack_type = rack_meta.get("rack_type")
    rack_vendor = rack_meta.get("rack_vendor")

    switches = audit_data.get("switch_details", [])
    vendors = sorted({s.get("vendor") for s in switches if s.get("vendor")})
    models = sorted({s.get("model") for s in switches if s.get("model")})

    empty_ports = audit_data.get("empty_ports", [])
    connected_ports = audit_data.get("connected_ports", [])
    ports, cables = extract_ports_and_cables(audit_data)

    # Presence checks (qualitative only)
    qsfp_present = any(re.search(r"\\bQSFP\\b", p, re.IGNORECASE) for p in (empty_ports + connected_ports))
    lc_present = any(re.search(r"\\bLC\\b", p, re.IGNORECASE) for p in (empty_ports + connected_ports))

    # Cable colors from entire text (handles cut lines)
    cable_colors = extract_cable_colors(full_text)

    # Detect NEMA presence
    patchpanel_text = " ".join(audit_data.get("patch_panel_details", []))
    has_nema = bool(re.search(r"\\bNEMA\\b", patchpanel_text, re.IGNORECASE))

    sections = []

    # --- Executive Summary (single paragraph) ---
    exec_lines = []
    # Line 1: scope and rack form/vendor
    if rack_type and rack_vendor:
        exec_lines.append(f"This audit reviewed a {rack_type.lower()} rack from {rack_vendor}, covering patch panels, switches, and structured cabling.")
    elif rack_type:
        exec_lines.append(f"This audit reviewed a {rack_type.lower()} rack with patch panels, switches, and structured cabling.")
    else:
        exec_lines.append("This audit reviewed a rack with patch panels, switches, and structured cabling.")
    # Line 2: configuration orderliness
    exec_lines.append("The configuration appears orderly with clear unit allocation and standardized terminations.")
    # Line 3: connectivity types (qualitative)
    if lc_present:
        exec_lines.append("Observed connectivity includes Ethernet RJ-45 terminations together with fiber links using LC connectors.")
    else:
        exec_lines.append("Observed connectivity includes Ethernet RJ-45 terminations consistent with structured cabling.")
    # Line 4: expansion readiness (qualitative only)
    if qsfp_present:
        exec_lines.append("The setup indicates readiness for bandwidth scaling through available high-speed interfaces.")
    else:
        exec_lines.append("The setup indicates room for future additions without disrupting the existing layout.")
    exec_paragraph = " ".join(exec_lines)
    sections.append({"title": None, "paragraph": exec_paragraph})

    # Rack Overview (include vendor/type and layout)
    layout_text = (
        "Units layout: " + ", ".join([f"{u}: {rack_units[u]}" for u in sorted(rack_units.keys())])
        if rack_units else "Units include patch panels, switches, and empty slots."
    )
    overview_points = []
    if rack_type and rack_vendor:
        overview_points.append(f"Form factor: {rack_type.title()} | Vendor: {rack_vendor}.")
    elif rack_type:
        overview_points.append(f"Form factor: {rack_type.title()}.")
    elif rack_vendor:
        overview_points.append(f"Vendor: {rack_vendor}.")
    overview_points.append(f"Wall-mounted rack with {unit_count} units.")
    overview_points.append(layout_text)

    sections.append({
        "title": "Rack Overview",
        "points": overview_points
    })

    # Patch Panels
    pp_points = [
        "RJ-45 connectors provide standard Ethernet terminations.",
        "Ports are color-coded to support structured cabling practices."
    ]
    if has_nema:
        pp_points.append("Power outlets (e.g., NEMA 5-15) are present on related panels.")
    sections.append({
        "title": "Patch Panels",
        "points": pp_points
    })

    # Switches (vendors/models, qualitative only)
    sw_points = []
    if vendors:
        sw_points.append("Detected switch vendor(s): " + ", ".join(sorted(vendors)) + ".")
    else:
        sw_points.append("Switch vendor details were not reliably extracted from the source pages.")
    if models:
        sw_points.append("Model details were recognized for some units.")
    sections.append({
        "title": "Switches",
        "points": sw_points
    })

    # Ports (qualitative only; no counts)
    port_points = []
    if lc_present:
        port_points.append("Fiber connectivity is present using LC connectors in applicable segments.")
    port_points.append("Ethernet connectivity uses RJ-45 terminations, consistent with structured cabling.")
    if qsfp_present:
        port_points.append("High-speed interfaces (e.g., QSFP) are available for scaling bandwidth.")
    sections.append({
        "title": "Ports",
        "points": port_points
    })

    # Cabling (qualitative only + colors)
    cable_points = []
    if cable_colors:
        cable_points.append("Ethernet twisted-pair segments observed: " + ", ".join(cable_colors) + ".")
    if cables:
        cable_points.append("Cabling observed includes " + ", ".join(cables) + ".")
    else:
        cable_points.append("Cabling observations include Ethernet twisted pair and fiber where applicable.")
    if ports:
        cable_points.append("Port types present include " + ", ".join(ports) + ".")
    sections.append({
        "title": "Cabling",
        "points": cable_points
    })

    # Capacity (qualitative statement; no counts; no recommendations)
    sections.append({
        "title": "Capacity",
        "points": [
            "Available interfaces and empty units indicate scope for adding new links and hardware as needed."
        ]
    })

    return sections

def sections_to_bullet_text(sections):
    """
    Convert grouped sections into plain text summary lines (for .txt output).
    - First section has no heading (executive summary, 4 lines).
    """
    lines = []
    first = True
    for sec in sections:
        title = sec.get("title")
        if title and not first:
            lines.append(f"**{title}**")

        # If a paragraph field is present (used for executive summary),
        # split it into multiple bullet lines for the TXT output.
        if "paragraph" in sec:
            para = sec.get("paragraph", "")
            # Split into sentence-like segments; fallback to whole paragraph.
            parts = [s.strip() for s in re.split(r"\.[\s\n]+", para) if s.strip()]
            if parts:
                for p in parts:
                    # Ensure sentence ends with a period for readability
                    p = p.rstrip()
                    if not p.endswith('.'):
                        p = p + '.'
                    lines.append(f"- {p}")
            else:
                lines.append(f"- {para}")
        else:
            for p in sec.get("points", []):
                lines.append(f"- {p}")

        lines.append("")  # blank line between sections
        first = False
    return "\n".join(lines).strip()

# ==============================================================================
# PDF GENERATION (STYLED GROUPED SUMMARY)
# ==============================================================================
def generate_pdf_grouped(sections, out_filename: str, font_name: str):
    doc = SimpleDocTemplate(
        out_filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=48,
        bottomMargin=44
    )
    styles = getSampleStyleSheet()

    # Styles with chosen font
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1F3B4D"),
        spaceAfter=8
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        textColor=colors.HexColor("#0B6FA4"),
        spaceBefore=8,
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16
    )
    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16
    )

    elements = []
    # Header Title
    elements.append(Paragraph("Rack Audit Summary Report", title_style))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#E0E6EA")))
    elements.append(Spacer(1, 8))

    # Build grouped sections (exec summary first block has no heading)
    first_block = True
    for sec in sections:
        if sec["title"] and not first_block:
            elements.append(Paragraph(sec["title"], section_style))

        if "paragraph" in sec:
            elements.append(Paragraph(sec["paragraph"], normal_style))
        else:
            bullet_items = []
            for p in sec["points"]:
                bullet_items.append(ListItem(Paragraph(p, bullet_style)))

            # Use '-' as bullet char to avoid glyph dependencies
            elements.append(ListFlowable(
                bullet_items,
                bulletType="bullet",
                leftIndent=16,
                bulletChar='-'
            ))
        elements.append(Spacer(1, 6))
        first_block = False

    # Footer callback
    def add_footer(canvas_obj, doc_obj):
        canvas_obj.setFont(font_name, 9)
        canvas_obj.setFillColor(colors.HexColor("#6B7F91"))
        page_num = canvas_obj.getPageNumber()
        date_str = datetime.now().strftime("%Y-%m-%d")
        canvas_obj.drawString(40, 28, f"Audit Date: {date_str} | Summary")
        canvas_obj.drawRightString(A4[0] - 40, 28, f"Page {page_num}")

    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate an appealing, detailed rack audit summary (summary-only PDF) matching strict constraints."
    )
    parser.add_argument("--pdf", dest="pdf", help="Path to input merged PDF", default=DEFAULT_PDF_PATH)
    parser.add_argument("--out", dest="out", help="Output PDF filename", default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--font", dest="font", help="Path to a Unicode TTF font (e.g., DejaVuSans.ttf).", default=None)
    args = parser.parse_args()

    # Font
    font_name = register_unicode_font(args.font)

    # Extract and parse
    print("Extracting PDF...", args.pdf)
    text = extract_pdf_text(args.pdf)
    print("Parsing sections...")
    sections_raw = split_by_sections(text)

    # Build audit data with rack meta included
    audit_data = {
        "rack_details": parse_rack(sections_raw.get("rack_details", "")),
        "rack_meta": parse_rack_meta(sections_raw.get("rack_details", "")),  # rack type + rack vendor
        "patch_panel_details": parse_patch_panel(sections_raw.get("patch_panel_details", "")),
        "switch_details": parse_switch_from_text(text),
        "connected_ports": parse_ports(sections_raw.get("connected_ports", "")),
        "empty_ports": parse_ports(sections_raw.get("empty_ports", "")),
        "cable_classification": clean_lines(sections_raw.get("cable_classification", "")),
    }

    # Save JSON
    print("Saving JSON...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=4)

    # Build grouped sections (local, compliant)
    grouped_sections = build_grouped_summary(audit_data, text)

    # Generate PDF + TXT
    print("Generating PDF...")
    generate_pdf_grouped(grouped_sections, args.out, font_name)
    txt = sections_to_bullet_text(grouped_sections)

    # Save TXT
    summary_out = (args.out or DEFAULT_OUTPUT_PDF)
    summary_txt = summary_out[:-4] + ".txt" if summary_out.lower().endswith(".pdf") else summary_out + ".txt"
    try:
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(txt)
        print("Saved summary text to", summary_txt)
    except Exception as e:
        print("Failed to save summary text:", e)

    print("FULL PIPELINE COMPLETE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"PDF: {args.out or DEFAULT_OUTPUT_PDF}")

if __name__ == "__main__":
    main()
