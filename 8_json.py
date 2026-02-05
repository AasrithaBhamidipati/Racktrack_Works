# -*- coding: utf-8 -*-
"""
Structured Rack Audit JSON Generator

Generates a clean, hierarchical JSON structure:
- Rack information at the top level
- Units as array with components and ports
- Cables as separate top-level array
- Optimized for analysis and reporting

Usage:
------
python 8_json_structured.py --input "jobs_output/job_XXX/output/Results" --out "audit_report.json"
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import sys
import warnings
try:
    import pandas as pd
except Exception:
    pd = None

class RackAuditJSONGenerator:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.rack_data = {
            "type": "Network Rack",
            "vendor": "Generic / Unknown",
            "report_generated": datetime.now().strftime("%Y-%m-%d"),
            "summary": "",
            "units": []
        }
        self.unit_map = {}
        self.port_id_counter = 0  # Global counter for unique port IDs
        # switch description mappings loaded from Excel files
        self.switch_desc_map = {}  # (vendor_lower, model_lower) -> description
        self.switch_model_map = {}  # model_lower -> description
        self._load_switch_descriptions()
    
    def extract_text_from_table(self, table) -> dict:
        """Extract key-value pairs from HTML table"""
        data = {}
        if not table:
            return data
        
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text().strip()
                value = cells[1].get_text().strip()
                if key and value:
                    data[key.lower().replace(" ", "_")] = value
        return data

    def _clean_meta_value(self, value: str, field: str = "") -> str | None:
        """Return cleaned metadata value or None if it looks like a header/placeholder.

        Filters out values like 'Switch Model', 'Description', 'Detected OCR',
        or bare header words such as 'model', 'vendor', and common placeholders.
        """
        if not value:
            return None
        v = value.strip()
        low = v.lower()

        # blacklist of known bad tokens or exact placeholders
        bad_tokens = {
            "switch model", "description", "detected ocr", "detected", "ocr",
            "model", "vendor", "switch", "device", "device info",
            "not detected", "unknown", "none", "n/a", "na"
        }

        # If value matches any bad token exactly, reject it
        if low in bad_tokens:
            return None

        # If the value is short and looks like a header (one or two words like 'model description')
        if len(low.split()) <= 2 and any(tok in low for tok in ["model", "description", "vendor", "detected", "ocr"]):
            return None

        # If it looks like it's all uppercase header-like words, reject
        if v.isupper() and len(v.split()) <= 3:
            return None

        # Otherwise return cleaned string
        return v

    def _normalize_description(self, text: str) -> str:
        """Normalize description text: replace newlines with spaces and collapse whitespace."""
        if not text:
            return text
        s = text.replace('\r', ' ').replace('\n', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _load_switch_descriptions(self):
        """Load switch descriptions from Description_excel/switch_excel folder.

        Each vendor file is expected to contain model and description columns.
        """
        if pd is None:
            print("[SwitchDesc] pandas not available; skipping switch description loading")
            return

        base = Path(__file__).resolve().parent.parent
        switch_dir = base / "Description_excel" / "switch_excel"
        if not switch_dir.exists():
            return

        for f in switch_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in [".xlsx", ".xls", ".csv"]:
                continue
            vendor = f.stem
            try:
                if f.suffix.lower() in [".csv"]:
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
            except Exception as e:
                warnings.warn(f"Failed to read switch excel {f}: {e}")
                continue

            # normalize column names
            cols = {c.lower(): c for c in df.columns}
            model_col = None
            desc_col = None
            for k in cols:
                if 'model' in k:
                    model_col = cols[k]
                if 'desc' in k or 'description' in k:
                    desc_col = cols[k]

            if model_col is None:
                # try first column
                model_col = df.columns[0] if len(df.columns) > 0 else None
            if desc_col is None:
                # try second column
                desc_col = df.columns[1] if len(df.columns) > 1 else None

            if model_col is None:
                continue

            for _, row in df.iterrows():
                try:
                    m = str(row[model_col]).strip()
                    d = str(row[desc_col]).strip() if desc_col and not pd.isna(row[desc_col]) else ''
                except Exception:
                    continue
                if not m or m.lower() in ['nan', 'none']:
                    continue
                # normalize model key to be alphanumeric + underscores
                mk = re.sub(r'[^a-z0-9]+', '_', m.lower()).strip('_')
                key = (vendor.lower(), mk)
                if d and d.lower() not in ['nan', 'none']:
                    nd = self._normalize_description(d)
                    self.switch_desc_map[key] = nd
                    if mk not in self.switch_model_map:
                        self.switch_model_map[mk] = nd
                else:
                    # at least populate model map with vendor as fallback
                    if mk not in self.switch_model_map:
                        self.switch_model_map[mk] = ''

    def _lookup_switch_description(self, vendor: str | None, model: str | None) -> str | None:
        """Return a description for given vendor/model using loaded excel maps."""
        if not (vendor or model):
            return None
        # normalize model
        mk = None
        if model:
            mk = re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')
        if vendor and mk:
            d = self.switch_desc_map.get((vendor.lower(), mk))
            if d:
                return d
        if mk:
            d = self.switch_model_map.get(mk)
            if d:
                return d if d else None
        # fallback: try to find any model entry containing model substring
        if mk:
            for k, v in self.switch_model_map.items():
                if mk in k and v:
                    return v
        return None

    def _normalize_description(self, desc: str) -> str:
        """Normalize description by replacing newlines with spaces."""
        if not desc:
            return desc
        return desc.replace('\n', ' ').replace('\r', '').strip()

    def _guess_vendor_model_description(self, texts: list) -> tuple:
        """Heuristic to guess vendor, model, and description from a list of text values."""
        vendor = None
        model = None
        description = None

        known_vendors = [
            'cisco','juniper','hpe','hp','aruba','dell','dlink','netgear','tp-link','ubiquiti','mikrotik','cumulus','brocade','zte'
        ]

        # normalize texts
        cleaned = [t.strip() for t in texts if t and t.strip()]
        lowered = [t.lower() for t in cleaned]

        # find vendor by known vendor names
        for kv in known_vendors:
            for i, low in enumerate(lowered):
                if kv in low:
                    vendor = cleaned[i]
                    break
            if vendor:
                break

        # find model by pattern (contains digits and letters, hyphen or underscore)
        model_re = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-_/]{2,}')
        for i, t in enumerate(cleaned):
            if model_re.search(t) and len(t) <= 60:
                # avoid picking long description-like texts
                if not vendor or cleaned[i] != vendor:
                    model = cleaned[i]
                    break

        # pick description as the longest remaining string
        longest = ""
        for t in cleaned:
            low = t.lower()
            if t == vendor or t == model:
                continue
            if len(t) > len(longest):
                longest = t
        if longest and len(longest) > 8:
            description = longest

        return vendor, model, description

    def _guess_patchpanel_model(self, text: str) -> str | None:
        """Heuristic to detect patch panel model names from free text."""
        if not text:
            return None
        t = text.lower()
        # common patch panel model tokens and their normalized forms
        tokens = {
            r'rj[-_\s]?45': 'RJ_45',
            r'lc\s*upc': 'LC_UPC',
            r'sc\s*upc': 'SC_UPC',
            r'mpo': 'MPO',
            r'st\b': 'ST',
            r'keystone': 'Keystone',
            r'patch\s*panel': None
        }

        for patt, norm in tokens.items():
            if re.search(patt, t, flags=re.I):
                if norm:
                    return norm
                # if token is generic like 'patch panel', keep searching

        # Also look for uppercase-like model strings with numbers (e.g., 24 port, 48-port)
        m = re.search(r'(\b\d{1,2}[- ]?port\b)|(\b\d{1,2}p\b)', t, flags=re.I)
        if m:
            return m.group(0).replace(' ', '_')

        return None
    
    def parse_rack_report(self):
        """Parse rack_report.html"""
        rack_file = self.results_dir / "rack_report.html"
        if not rack_file.exists():
            print(f"⚠️ rack_report.html not found")
            return
        
        try:
            with open(rack_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract from device-info section
            device_info = soup.find('div', class_='device-info')
            if device_info:
                text = device_info.get_text()
                
                # Extract rack type
                type_match = re.search(r'[Tt]ype[:\s]+([^\n]+)', text)
                if type_match:
                    rack_type = type_match.group(1).strip().lower()
                    if "enclosed" in rack_type:
                        self.rack_data["type"] = "enclosed"
                    elif "openframe" in rack_type or "open frame" in rack_type:
                        self.rack_data["type"] = "openframe"
                    elif "wall" in rack_type:
                        self.rack_data["type"] = "wall_mounted"
                
                # Extract vendor
                vendor_match = re.search(r'[Vv]endor[:\s]+([^\n]+)', text)
                if vendor_match:
                    vendor = vendor_match.group(1).strip()
                    if vendor and vendor != "Not Detected" and len(vendor) > 2:
                        self.rack_data["vendor"] = vendor

            # Fallbacks: some reports include a table with "Rack Type & Vendor" or separate columns.
            # Check first table row/columns for type/vendor info if not found above.
            try:
                table = soup.find('table')
                if table:
                    rows = table.find_all('tr')
                    if len(rows) > 1:
                        headers = [h.get_text().strip().lower() for h in rows[0].find_all(['th', 'td'])]
                        cells = rows[1].find_all(['td', 'th'])
                        for idx, h in enumerate(headers):
                            if 'rack type' in h or 'rack type & vendor' in h or (h.strip() == 'type'):
                                if idx < len(cells):
                                    cell_text = cells[idx].get_text().strip()
                                    # common separators: | - ; , /
                                    parts = [p.strip() for p in re.split(r'[\|\-;,/]+', cell_text) if p.strip()]
                                    if parts:
                                        maybe_type = parts[0].lower()
                                        if 'enclosed' in maybe_type:
                                            self.rack_data['type'] = 'enclosed'
                                        elif 'openframe' in maybe_type or 'open frame' in maybe_type or 'open' in maybe_type:
                                            self.rack_data['type'] = 'openframe'
                                        elif 'wall' in maybe_type:
                                            self.rack_data['type'] = 'wall_mounted'
                                        # vendor may follow
                                        if len(parts) > 1:
                                            maybe_vendor = parts[1]
                                            if maybe_vendor and maybe_vendor.lower() not in ['not detected', 'unknown', '']:
                                                self.rack_data['vendor'] = maybe_vendor
                            if 'vendor' in h and 'vendor' not in self.rack_data:
                                if idx < len(cells):
                                    v = cells[idx].get_text().strip()
                                    if v and v.lower() not in ['not detected', 'unknown', '']:
                                        self.rack_data['vendor'] = v
            except Exception:
                pass
            
            print(f"Parsed rack report: type={self.rack_data['type']}, vendor={self.rack_data['vendor']}")
        
        except Exception as e:
            print(f"⚠️ Error parsing rack report: {e}")
    
    def parse_unit_reports(self):
        """Parse all unit_X_report.html files"""
        unit_files = sorted(self.results_dir.glob("unit*_report.html"))
        
        for unit_file in unit_files:
            try:
                # Extract unit number from filename
                match = re.search(r'unit(\d+)', unit_file.name)
                if not match:
                    continue
                
                unit_id = int(match.group(1))
                
                with open(unit_file, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                
                # Extract device type from badge
                device_badge = soup.find('div', class_='device-badge')
                device_type = device_badge.get_text().strip().lower() if device_badge else "unknown"
                
                # Build unit structure with new format
                unit = {
                    "unit_id": f"unit{unit_id}",
                    "device_type": device_type,
                    "vendor": None,
                    "model": None,
                    "description": None,
                    "connected_ports": [],
                    "empty_ports": []
                }

                
                # Extract device info from device-info div
                device_info_div = soup.find('div', class_='device-info')
                if device_info_div:
                    # Look for tables within device-info (Switch Analysis Report, etc.)
                    tables = device_info_div.find_all('table')
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        if len(rows) > 1:  # Has header + data
                            headers = [h.get_text().strip().lower() for h in rows[0].find_all(['th', 'td'])]
                            
                            # Look for vendor, model, description columns (improved matching)
                            vendor_idx = next((i for i, h in enumerate(headers) if 'vendor' in h), -1)
                            model_idx = next((i for i, h in enumerate(headers) if 'model' in h), -1)
                            desc_idx = next((i for i, h in enumerate(headers) if 'description' in h), -1)
                            
                            # Extract from first data row
                            if len(rows) > 1:
                                data_row = rows[1]
                                cells = data_row.find_all(['td', 'th'])
                                
                                # Extract vendor
                                if vendor_idx >= 0 and vendor_idx < len(cells):
                                    vendor = cells[vendor_idx].get_text().strip()
                                    cleaned_vendor = None
                                    if device_type.lower() not in ["patch panel", "patchpanel"]:
                                        cleaned_vendor = self._clean_meta_value(vendor, "vendor")
                                        if cleaned_vendor:
                                            unit["vendor"] = cleaned_vendor
                                
                                # Extract model (don't filter with _clean_meta_value for models)
                                if model_idx >= 0 and model_idx < len(cells):
                                    model = cells[model_idx].get_text().strip()
                                    if model and model.lower() not in ["unknown", "not detected", "model", "switch model"]:
                                        unit["model"] = model
                                
                                # Extract description
                                if desc_idx >= 0 and desc_idx < len(cells):
                                    desc = cells[desc_idx].get_text().strip()
                                    cleaned_desc = self._clean_meta_value(desc, "description")
                                    if cleaned_desc and len(cleaned_desc) > 2:
                                        unit["description"] = self._normalize_description(cleaned_desc)

                                # If any of vendor/model/description are still missing, try heuristic guessing
                                if device_type.lower() == 'switch':
                                    cell_texts = [c.get_text().strip() for c in cells]
                                    guessed_vendor, guessed_model, guessed_desc = self._guess_vendor_model_description(cell_texts)
                                    if 'vendor' not in unit and guessed_vendor:
                                        unit['vendor'] = guessed_vendor
                                    if 'model' not in unit and guessed_model:
                                        unit['model'] = guessed_model
                                    if 'description' not in unit and guessed_desc:
                                        unit['description'] = guessed_desc
                
                # Also try to extract from device-info raw text
                if device_info_div:
                    device_text = device_info_div.get_text()
                    
                    # Try to extract vendor (only if not already found and not a patch panel)
                    if "vendor" not in unit and device_type.lower() not in ["patch panel", "patchpanel"]:
                        vendor_match = re.search(r'[Vv]endor[:\s]+([^\n,]+)', device_text)
                        if vendor_match:
                            vendor = vendor_match.group(1).strip()
                            cleaned_vendor = self._clean_meta_value(vendor, "vendor")
                            if cleaned_vendor:
                                unit["component"]["vendor"] = cleaned_vendor
                    
                    # Try to extract model (only if not already found)
                    if "model" not in unit:
                        model_match = re.search(r'[Mm]odel[:\s]+([^\n,]+)', device_text)
                        if model_match:
                            model = model_match.group(1).strip()
                            cleaned_model = self._clean_meta_value(model, "model")
                            if cleaned_model:
                                unit["model"] = cleaned_model
                    
                    # Try to extract description
                    if "description" not in unit:
                        desc_match = re.search(r'[Dd]escription[:\s]+([^\n]+)', device_text)
                        if desc_match:
                            desc_val = desc_match.group(1).strip()
                            cleaned_desc = self._clean_meta_value(desc_val, "description")
                            if cleaned_desc:
                                unit["description"] = self._normalize_description(cleaned_desc)

                # For patch panels, if model is missing, try to detect from device text or description
                if 'patch' in device_type.lower():
                    model_present = 'model' in unit and unit.get('model')
                    if not model_present:
                        combined_text = ''
                        try:
                            combined_text += device_info_div.get_text() if device_info_div else ''
                        except Exception:
                            pass
                        combined_text += ' ' + unit.get('description', '')
                        guessed = self._guess_patchpanel_model(combined_text)
                        if guessed:
                            unit['model'] = guessed
                
                # Parse connected and empty ports
                sections = soup.find_all('div', class_='section')
                for section in sections:
                    title = section.find('h2', class_='section-title')
                    if not title:
                        continue
                    
                    section_text = title.get_text().strip()
                    
                    if '✅' in section_text or 'Connected' in section_text:
                        # Parse connected ports
                        table = section.find('table')
                        if table:
                            rows = table.find_all('tr')[1:]  # Skip header
                            for row in rows:
                                cells = row.find_all('td')
                                if len(cells) >= 6:
                                    # Extract from table columns: Cable, Description, Usage, Port, Port Desc, Other Port
                                    cable = cells[1].get_text().strip() if len(cells) > 1 else "RJ_45"
                                    cable_description = cells[2].get_text().strip() if len(cells) > 2 else ""
                                    usage = cells[3].get_text().strip() if len(cells) > 3 else "Ethernet networking"
                                    port = cells[4].get_text().strip() if len(cells) > 4 else ""
                                    port_description = cells[5].get_text().strip() if len(cells) > 5 else ""
                                    other_port = cells[6].get_text().strip() if len(cells) > 6 else ""
                                    
                                    # Skip ports with N/A cable_description and usage (non-networking ports like HDMI, IEC)
                                    if cable_description.lower() == "n/a" and usage.lower() == "n/a":
                                        continue
                                    
                                    self.port_id_counter += 1
                                    connected_port = {
                                        "port_id": f"port_{self.port_id_counter}",
                                        "port": port,
                                        "port_description": port_description,
                                        "other_port": other_port,
                                        "cable": cable,
                                        "cable_description": cable_description,
                                        "usage": usage
                                    }
                                    unit["connected_ports"].append(connected_port)
                    
                    elif '⭕' in section_text or 'Empty' in section_text:
                        # Parse empty ports
                        table = section.find('table')
                        if table:
                            rows = table.find_all('tr')[1:]  # Skip header
                            for row in rows:
                                cells = row.find_all('td')
                                if len(cells) >= 1:
                                    port_type = "Ethernet"
                                    port_description = "Eight-pin Ethernet connector used for wired LANs, supporting speeds from 10 Mbps to 10 Gbps depending on cable type."
                                    
                                    if "SFP" in cells[0].get_text():
                                        port_type = "SFP"
                                        port_description = "Small Form-factor Pluggable transceiver used for high-speed fiber optic connections in telecommunications and data center networks."
                                    elif "QSFP" in cells[0].get_text():
                                        port_type = "QSFP28"
                                        port_description = "Quad Small Form-factor Pluggable 28 connector supporting 100 Gbps high-speed data transmission over fiber optic cables."
                                    
                                    self.port_id_counter += 1
                                    empty_port = {
                                        "port_id": f"port_{self.port_id_counter}",
                                        "type": port_type,
                                        "description": port_description
                                    }
                                    unit["empty_ports"].append(empty_port)
                
                # For switches, prefer description from Excel lookup when available
                if device_type.lower() == 'switch':
                    v = unit.get('vendor')
                    m = unit.get('model')
                    desc_lookup = self._lookup_switch_description(v, m)
                    if desc_lookup:
                        unit['description'] = self._normalize_description(desc_lookup)

                # Remove None values from unit, but keep model and vendor fields only if they have values
                unit_clean = {}
                for k, v in unit.items():
                    # Skip None values
                    if v is None:
                        # Only include vendor/model for switches if they have non-null values
                        if k in ["model", "vendor"] and device_type.lower() == "switch" and v is not None:
                            unit_clean[k] = v
                        else:
                            continue
                    else:
                        unit_clean[k] = v
                
                self.rack_data["units"].append(unit_clean)
                vendor = unit_clean.get("vendor", "Unknown")
                model = unit_clean.get("model", "Unknown")
                connected_count = len(unit_clean.get('connected_ports', []))
                empty_count = len(unit_clean.get('empty_ports', []))
                print(f"Parsed unit {unit_id}: {device_type} | Vendor: {vendor} | Model: {model} | Connected: {connected_count} | Empty: {empty_count}")
            
            except Exception as e:
                print(f"⚠️ Error parsing {unit_file.name}: {e}")
    

    
    def generate(self) -> dict:
        """Generate the complete structured JSON"""
        print(f"\n[JSON Generator] Parsing reports from: {self.results_dir}")
        
        self.parse_rack_report()
        self.parse_unit_reports()
        
        # Generate summary
        units_summary = []
        for unit in self.rack_data.get("units", []):
            device_type = unit.get("device_type", "Device")
            vendor = unit.get("vendor", "")
            if vendor:
                units_summary.append(f"{vendor} {device_type}")
            else:
                units_summary.append(device_type)
        
        devices = ", ".join(units_summary) if units_summary else "network devices"
        self.rack_data["summary"] = f"Rack analysis detected a populated network rack containing {devices}."
        
        return self.rack_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured JSON from rack audit HTML reports"
    )
    parser.add_argument("--input", required=True, help="Input directory with HTML reports")
    parser.add_argument("--out", default=None, help="Output JSON file (optional). Defaults to <input>/audit_report.json")
    
    args, _ = parser.parse_known_args()
    
    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"❌ Error: {results_dir} does not exist")
        sys.exit(1)
    
    # Default output JSON to Results/audit_report.json when not provided
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = results_dir / "audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator = RackAuditJSONGenerator(results_dir)
    audit_json = generator.generate()
    
    # Write JSON
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON saved to: {out_path}")
    
    # Check for merged PDF in the same Results folder
    merged_pdf = results_dir / "Merged_Result.pdf"
    if merged_pdf.exists():
        print(f"Merged PDF found at: {merged_pdf}")
    else:
        print(f"Merged PDF not found at expected location: {merged_pdf}")

    # Print summary
    print(f"\nSUMMARY:")
    print(f"  - Rack Type: {audit_json.get('type', 'unknown')}")
    print(f"  - Rack Vendor: {audit_json.get('vendor', 'Not Detected')}")
    print(f"  - Units: {len(audit_json.get('units', []))}")
    
    total_connections = sum(len(u.get('connected_ports', [])) for u in audit_json.get('units', []))
    total_empty = sum(len(u.get('empty_ports', [])) for u in audit_json.get('units', []))
    print(f"  - Total Connected Ports: {total_connections}")
    print(f"  - Total Empty Ports: {total_empty}")



if __name__ == "__main__":
    main()
