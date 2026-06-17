# -*- coding: utf-8 -*-
import os, re, base64, subprocess, sys, mimetypes

def ensure(pkg, imp=None):
    try: __import__(imp or pkg)
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
ensure("markdown")
import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
md_path = os.path.join(ROOT, "RAPPORT.md")
html_path = os.path.join(ROOT, "RAPPORT.html")

with open(md_path, encoding="utf-8") as f:
    text = f.read()

# Convertit le markdown -> HTML (tables, code, table des matieres)
body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list"])

# Embarque chaque image en base64 pour un fichier auto-portant
def embed(m):
    src = m.group(1)
    path = os.path.normpath(os.path.join(ROOT, src))
    if not os.path.exists(path):
        return m.group(0)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as im:
        data = base64.b64encode(im.read()).decode()
    return f'src="data:{mime};base64,{data}"'

body = re.sub(r'src="([^"]+)"', embed, body)

# Rendu MathJax pour les formules LaTeX ($...$)
html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>Rapport - Projet Deep Learning EMSI</title>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 880px; margin: 40px auto;
         padding: 0 24px; color: #1a1a1a; line-height: 1.6; }}
  h1, h2, h3 {{ font-family: 'Segoe UI', Arial, sans-serif; color: #0b3d2e; line-height: 1.25; }}
  h1 {{ border-bottom: 3px solid #0b3d2e; padding-bottom: 8px; }}
  h2 {{ border-bottom: 1px solid #cfd8d3; padding-bottom: 4px; margin-top: 2em; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 16px auto;
        box-shadow: 0 1px 6px rgba(0,0,0,.15); border-radius: 4px; }}
  table {{ border-collapse: collapse; margin: 16px auto; font-family: 'Segoe UI', Arial, sans-serif;
          font-size: .94em; }}
  th, td {{ border: 1px solid #b9c4be; padding: 6px 12px; text-align: center; }}
  th {{ background: #eef3f0; }}
  blockquote {{ background: #f3f7f5; border-left: 4px solid #0b3d2e; margin: 16px 0;
               padding: 10px 18px; border-radius: 0 4px 4px 0; }}
  code {{ background: #eef1ef; padding: 1px 5px; border-radius: 3px;
         font-family: Consolas, monospace; font-size: .92em; }}
  @media print {{ body {{ max-width: 100%; }} a {{ color: inherit; text-decoration: none; }} }}
</style></head>
<body>
{body}
</body></html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print("HTML auto-portant ->", html_path, "| taille:", round(os.path.getsize(html_path)/1024, 1), "Ko")
