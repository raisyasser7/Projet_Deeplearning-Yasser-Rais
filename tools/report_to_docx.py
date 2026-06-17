# -*- coding: utf-8 -*-
"""Convertit RAPPORT.md -> RAPPORT.docx (Word) avec images, tableaux et formules."""
import os, pypandoc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
md = os.path.join(ROOT, "RAPPORT.md")
docx = os.path.join(ROOT, "RAPPORT.docx")

# On execute depuis ROOT pour que les chemins d'images relatifs se resolvent.
def _convert(target):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        pypandoc.convert_file(
            md, "docx", outputfile=target,
            format="markdown+pipe_tables+tex_math_dollars",
            extra_args=["--resource-path", ROOT],
        )
    finally:
        os.chdir(cwd)

try:
    _convert(docx)
    out = docx
except RuntimeError as e:
    if "ermission" in str(e):          # fichier ouvert dans Word -> nom alternatif
        out = os.path.join(ROOT, "RAPPORT_MAJ.docx")
        _convert(out)
        print("ATTENTION : RAPPORT.docx etait verrouille (ouvert dans Word).")
    else:
        raise

print("DOCX ->", out, "| taille:", round(os.path.getsize(out) / 1024, 1), "Ko")
