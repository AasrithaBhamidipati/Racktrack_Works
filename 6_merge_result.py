# -*- coding: utf-8 -*-

import os
import re
import shutil
import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
from PyPDF2 import PdfMerger
from playwright.sync_api import sync_playwright

# ======================================================
# CATEGORY ORDER
# ======================================================
CATEGORY_ORDER = ["rack", "switch", "patchpanel", "cable", "empty_port", "connected_port"]
ORDER_INDEX = {c: i for i, c in enumerate(CATEGORY_ORDER)}

def detect_category(filename: str) -> str:
    f = filename.lower().replace(" ", "_").replace("-", "_")
    if "rack" in f: return "rack"
    if "patchpanel" in f or "patch_panel" in f: return "patchpanel"
    if "switch" in f: return "switch"
    if "cable" in f: return "cable"
    if "connected_port" in f: return "connected_port"
    if "empty_port" in f or "ports_results" in f: return "empty_port"
    return "zzz_unknown"

def sort_by_category_then_name(files):
    return sorted(files, key=lambda n: (ORDER_INDEX.get(detect_category(n), 999), n.lower()))

# ======================================================
# HTML CONTENT CHECK (SKIP EMPTY REPORTS)
# ======================================================
def html_has_meaningful_text(path: Path) -> bool:
  try:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # If the HTML contains images (inline or img tags), treat as meaningful
    if re.search(r"<img\b|<picture\b|data:image/", txt, flags=re.I):
      return True
    # If there's a table, ensure it has at least one non-header data row with text
    if re.search(r"<table\b", txt, flags=re.I):
      rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", txt, flags=re.S | re.I)
      if len(rows) >= 2:
        # check that at least one non-header row contains text
        for tr in rows[1:]:
          inner = re.sub(r"<[^>]+>", "", tr).strip()
          if len(inner) > 3:
            return True
        return False
      return False
    # Otherwise strip tags and check remaining text length
    txt = re.sub(r"<[^>]+>", "", txt)
    return len(txt.strip()) > 30
  except Exception:
    return False

# ======================================================
# PDF MERGE (SAFE)
# ======================================================
def safe_write_merged_pdf(streams, out_path):
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    merger = PdfMerger()
    for s in streams:
        merger.append(s)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        merger.write(f)
    os.replace(tmp, out_path)
    merger.close()
    return True

# ======================================================
# CSS & JS — NO GAPS, NO IMAGES
# ======================================================
NO_GAPS_AND_HIDE_IMAGES_CSS = """
@page { margin: 0; }
html, body { margin:0 !important; padding:0 !important; }
* {
  margin:0 !important; padding:0 !important;
  page-break-before:auto !important;
  page-break-after:auto !important;
  page-break-inside:auto !important;
}
img, picture, svg, canvas { display:none !important; }
* { background:none !important; background-image:none !important; }
p, li, td, th { margin-bottom:2px !important; line-height:1.2 !important; }
h1,h2,h3,h4,h5,h6 { margin-bottom:4px !important; }
"""

REMOVE_EMPTY_SECTIONS_JS = r"""
(() => {
  const hasText = el => el && el.innerText && el.innerText.trim().length > 3;

  document.querySelectorAll('body *').forEach(el => {
    if (el.tagName === 'TABLE') {
      const rows = el.querySelectorAll('tr');
      if (![...rows].some(r => hasText(r))) el.remove();
    }
    if (['DIV','SECTION','ARTICLE'].includes(el.tagName)) {
      if (!hasText(el)) el.remove();
    }
    if (/H\d/.test(el.tagName)) {
      const next = el.nextElementSibling;
      if (!next || !hasText(next)) el.remove();
    }
  });
})();
"""

REMOVE_FILENAMES_JS = r"""
(() => {
  const isFile = t => /\.(png|jpe?g|bmp|svg|webp)$/i.test(t || '');
  document.querySelectorAll('td,th,p,li,span').forEach(el => {
    if (isFile(el.innerText)) el.remove();
    if ((el.innerText||'').toLowerCase().includes('confidence')) el.remove();
  });
})();
"""

TABLE_TO_BLOCKS_JS = r"""
(() => {
  document.querySelectorAll('table').forEach(table => {
    const headers = [];
    const hr = table.querySelector('tr');
    if (!hr) return;
    hr.querySelectorAll('th,td').forEach(c => headers.push(c.innerText.trim()));
    const rows = table.querySelectorAll('tr');
    const wrap = document.createElement('div');

    rows.forEach((r,i) => {
      if (i === 0) return;
      const b = document.createElement('div');
      r.querySelectorAll('td').forEach((c,ci) => {
        const p = document.createElement('p');
        p.innerHTML = `<strong>${headers[ci]||''}:</strong> ${c.innerText}`;
        b.appendChild(p);
      });
      wrap.appendChild(b);
    });
    table.replaceWith(wrap);
  });
})();
"""

# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Results")
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-no-images", action="store_true", help="Also produce the no-images merged PDF")
    args, _ = parser.parse_known_args()

    html_dir = Path(args.input)
    if not html_dir.exists():
        print("No Results folder found")
        return

    base_pdf = args.output or str(html_dir / "Merged_Result.pdf")
    no_img_pdf = str(Path(base_pdf).parent / "no images" / (Path(base_pdf).stem + "_no_images.pdf"))

    html_files = [p.name for p in html_dir.iterdir() if p.suffix == ".html"]
    html_files = sort_by_category_then_name(html_files)

    with_images = []
    no_images = []

    # By default, treat *_no_images.html (or filenames containing 'no_images' / 'no images')
    # as the NO-IMAGES variant and do NOT include them in the WITH-images merge.
    for f in html_files:
      p = html_dir / f
      if not html_has_meaningful_text(p):
        continue

      lname = f.lower()
      is_no_images_file = False
      if re.search(r"no[_\s-]?images", lname) or lname.endswith("_no_images.html"):
        is_no_images_file = True

      if is_no_images_file:
        # only add to no_images list when explicitly requested
        no_images.append(f)
      else:
        with_images.append(f)

    pdf_with = []
    pdf_no = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width":2000,"height":1200})

        # ---------- WITH IMAGES (combined into one continuous PDF to avoid gaps) ----------
        try:
          tmp_dir = html_dir / "_merge_tmp"
          if tmp_dir.exists():
            try:
              shutil.rmtree(tmp_dir)
            except Exception:
              pass
          tmp_dir.mkdir(parents=True, exist_ok=True)

          combined_sections = []
          for f in with_images:
            src_path = html_dir / f
            try:
              raw = src_path.read_text(encoding='utf-8')
            except Exception:
              raw = src_path.read_text(encoding='latin-1')

            m = re.search(r"<body[^>]*>(.*?)</body>", raw, flags=re.S | re.I)
            body = m.group(1) if m else raw

            base_dir = src_path.parent.resolve()
            # fix relative src attributes
            def fix_src(m):
              quote = m.group(1)
              url = m.group(2)
              if re.match(r'^(?:https?:|data:|file:|/)', url, flags=re.I):
                return f'src={quote}{url}{quote}'
              abs_url = f"file:///{(base_dir / url).as_posix()}"
              return f'src={quote}{abs_url}{quote}'

            body = re.sub(r'src=("|\')([^"\']+)("|\')', lambda mm: fix_src(mm), body)

            # fix relative href attributes
            def fix_href(m):
              quote = m.group(1)
              url = m.group(2)
              if re.match(r'^(?:https?:|data:|file:|/)', url, flags=re.I):
                return f'href={quote}{url}{quote}'
              abs_url = f"file:///{(base_dir / url).as_posix()}"
              return f'href={quote}{abs_url}{quote}'

            body = re.sub(r'href=("|\')([^"\']+)("|\')', lambda mm: fix_href(mm), body)

            # fix CSS url(...) occurrences (inline style or <style> blocks)
            def fix_css_url(m):
              quote = m.group(1) or ''
              url = m.group(2)
              if re.match(r'^(?:https?:|data:|file:|/)', url, flags=re.I):
                return f'url({quote}{url}{quote})'
              abs_url = f"file:///{(base_dir / url).as_posix()}"
              return f'url({quote}{abs_url}{quote})'

            body = re.sub(r'url\(("|\')?([^\)"\']+)("|\')?\)', lambda mm: fix_css_url(mm), body, flags=re.I)

            # fix srcset attributes (comma-separated list of urls with descriptors)
            def fix_srcset(m):
              quote = m.group(1)
              val = m.group(2)
              parts = [p.strip() for p in val.split(',') if p.strip()]
              out = []
              for p in parts:
                sp = p.split()
                url = sp[0]
                rest = ' '.join(sp[1:])
                if re.match(r'^(?:https?:|data:|file:|/)', url, flags=re.I):
                  out_url = url
                else:
                  out_url = f"file:///{(base_dir / url).as_posix()}"
                out.append((out_url + (" " + rest if rest else "")).strip())
              return f'srcset={quote}{", ".join(out)}{quote}'

            body = re.sub(r'srcset=("|\')([^"\']+)("|\')', lambda mm: fix_srcset(mm), body, flags=re.I)

            # helper: decide whether the body has meaningful text or images
            def body_has_meaningful_text(b: str) -> bool:
              try:
                # quick check for images
                if re.search(r"<img|<picture|<svg|<canvas", b, flags=re.I):
                  return True
                # if table exists, ensure it has at least one non-header data row
                if re.search(r"<table\b", b, flags=re.I):
                  rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", b, flags=re.S | re.I)
                  if len(rows) >= 2:
                    for tr in rows[1:]:
                      inner = re.sub(r"<[^>]+>", "", tr).strip()
                      if len(inner) > 3:
                        return True
                    return False
                  return False
                # strip tags and measure remaining text
                txt = re.sub(r"<[^>]+>", "", b).strip()
                return len(txt) > 30
              except Exception:
                return False

            # create a table row for this component only if its body contains meaningful content
            if not body_has_meaningful_text(body):
              # skip empty sections (e.g., empty switch or patchpanel reports)
              print(f"[6_merge_result] Skipping empty report: {f}")
              continue

            cat = detect_category(f)
            title = cat.replace('_', ' ').title()
            row = f"<tr class='component-row'><td class='comp-title'><strong>{title}</strong><br/><small>{f}</small></td><td class='comp-body'>{body}</td></tr>"
            combined_sections.append(row)

          # end for - build combined HTML table from accumulated rows
          combined_html = """
          <html>
          <head>
            <meta charset='utf-8'/>
            <style>
              @page { margin: 0; }
              html,body { margin:0; padding:0; }
              * { margin:0; padding:0; box-sizing:border-box; }
              table.components { width:100%; border-collapse:collapse; table-layout:fixed; }
              table.components td { vertical-align: top; padding:6px; border:1px solid #ccc; word-break:break-word; }
              .comp-title { width:220px; background:#f6f6f6; font-size:12px; padding:6px; }
              .comp-body { padding:6px; }
              .comp-body img { max-width:360px; max-height:400px; height:auto; display:block; margin:4px 0; }
              .component-row { page-break-inside: avoid; break-inside: avoid; }
              /* tighten spacing to avoid large gaps */
              p, h1, h2, h3, h4, h5, h6, ul, li { margin:0; padding:0; line-height:1.15; }
            </style>
          </head>
          <body>
          <table class='components'>""" + "\n".join(combined_sections) + "\n</table></body></html>"

          combined_path = tmp_dir / "__combined_with_images.html"
          combined_path.write_text(combined_html, encoding='utf-8')

          file_url = f"file:///{combined_path.resolve().as_posix()}"
          page.goto(file_url, wait_until='networkidle')
          page.emulate_media(media='print')
          try:
            page.add_style_tag(content='@page { margin: 0; } html,body { margin: 0; padding: 0; }')
          except Exception:
            pass

          # collapse empty component bodies (no text and no images) to avoid large white gaps
          try:
            page.evaluate(r"""
            () => {
              document.querySelectorAll('.comp-body').forEach(cb => {
                const hasText = (cb.innerText||'').trim().length > 2;
                const hasImg = !!cb.querySelector('img');
                if (!hasText && !hasImg) {
                  cb.style.display = 'none';
                }
              });
              // also remove any very large empty blocks
              const els = Array.from(document.querySelectorAll('div, section, article'));
              for (const el of els) {
                const h = el.getBoundingClientRect().height || 0;
                const txt = (el.innerText||'').trim();
                if (h > 800 && txt.length < 10) el.remove();
              }
            }
            """)
          except Exception:
            pass

          # compute total height
          try:
            total_height = page.evaluate('() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)')
            if not isinstance(total_height, (int, float)) or total_height <= 0:
              total_height = 1122
          except Exception:
            total_height = 1122

          # Remove very large empty blocks that cause big white gaps
          try:
            page.evaluate(r"""
            () => {
              const els = Array.from(document.querySelectorAll('div, section, article'));
              for (const el of els) {
                const h = el.getBoundingClientRect().height || 0;
                const txt = (el.innerText||'').trim();
                if (h > 800 && txt.length < 10) el.remove();
              }
            }
            """
            )
            # recompute height after cleanup
            try:
              total_height = page.evaluate('() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)')
            except Exception:
              pass
          except Exception:
            pass

          pdf = page.pdf(width='2000px', height=f'{int(total_height)}px', print_background=True, margin={'top':'0','bottom':'0','left':'0','right':'0'})
          pdf_with.append(BytesIO(pdf))
          print(f"Combined {len(with_images)} files into one continuous PDF (with images)")

        except Exception as e:
          print(f"[6_merge_result] combining with-images failed, falling back: {e}")
          for f in with_images:
            page.goto(f"file:///{(html_dir/f).resolve()}", wait_until="networkidle")
            page.emulate_media(media="print")
            pdf = page.pdf(width="420mm", height="297mm", print_background=True)
            pdf_with.append(BytesIO(pdf))
            page.goto("about:blank")

        # ---------- NO IMAGES (optional) ----------
        if args.include_no_images:
          for f in no_images:
            page.goto(f"file:///{(html_dir/f).resolve()}", wait_until="networkidle")
            page.emulate_media(media="print")

            page.add_style_tag(content=NO_GAPS_AND_HIDE_IMAGES_CSS)
            page.evaluate(REMOVE_FILENAMES_JS)
            page.evaluate(TABLE_TO_BLOCKS_JS)
            page.evaluate(REMOVE_EMPTY_SECTIONS_JS)

            pdf = page.pdf(
              width="420mm",
              height="297mm",
              margin={"top":"0","bottom":"0","left":"0","right":"0"},
              print_background=True
            )
            pdf_no.append(BytesIO(pdf))
            page.goto("about:blank")

        browser.close()

      # Persist the WITH-images merged PDF
    try:
        if len(pdf_with) > 0:
          safe_write_merged_pdf(pdf_with, base_pdf)
        else:
          print("[6_merge_result] No WITH-images PDFs were generated to merge.")
    except Exception as e:
        print(f"[6_merge_result] Failed to write WITH-images merged PDF: {e}")

      # Persist the NO-IMAGES merged PDF (if requested)
    if args.include_no_images:
        try:
          if len(pdf_no) > 0:
            safe_write_merged_pdf(pdf_no, no_img_pdf)
          else:
            print("[6_merge_result] No NO-IMAGES PDFs were generated to merge.")
        except Exception as e:
          print(f"[6_merge_result] Failed to write NO-IMAGES merged PDF: {e}")

    # Print ASCII-only messages to avoid Windows cp1252 encode errors
    try:
      print("MERGE COMPLETE")
      print("WITH images :", base_pdf)
      print("NO images  :", no_img_pdf)
    except UnicodeEncodeError:
      # fallback: encode to utf-8 and write to stdout.buffer
      import sys
      sys.stdout.buffer.write(f"MERGE COMPLETE\nWITH images : {base_pdf}\nNO images  : {no_img_pdf}\n".encode("utf-8"))

if __name__ == "__main__":
    main()
