"""Inspect image order in BTP_Report_Format.docx (title page layout)."""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
z = zipfile.ZipFile(ROOT / "BTP_Report_Format.docx")
xml = z.read("word/document.xml").decode("utf-8")
rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
embeds = re.findall(r'r:embed="([^"]+)"', xml)
print("embed order count:", len(embeds))
for i, e in enumerate(embeds[:12]):
    t = rid_to_target.get(e, "?")
    print(i, e, "->", t)
