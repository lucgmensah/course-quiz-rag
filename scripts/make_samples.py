"""Build the sample course PDFs in samples/ (original text, no licence issue).

Run: python scripts/make_samples.py
"""
import textwrap
from pathlib import Path

COURSES = {
    "photosynthese": [
        "La photosynthèse est le processus par lequel les plantes, les algues et certaines bactéries "
        "transforment l'énergie lumineuse en énergie chimique. Elle se déroule dans les chloroplastes, "
        "des organites riches en chlorophylle, le pigment vert qui absorbe surtout la lumière rouge "
        "et bleue. L'équation globale est : dioxyde de carbone plus eau, en présence de lumière, "
        "donnent du glucose et du dioxygène.",
        "La phase claire a lieu dans les thylakoïdes, des membranes à l'intérieur du chloroplaste. "
        "La lumière y excite la chlorophylle, ce qui permet de casser des molécules d'eau : c'est la "
        "photolyse de l'eau, qui libère du dioxygène. L'énergie captée est stockée sous forme d'ATP "
        "et de NADPH, deux molécules qui serviront à la phase suivante.",
        "La phase sombre, ou cycle de Calvin, se déroule dans le stroma du chloroplaste. Elle n'a pas "
        "besoin de lumière directement : elle utilise l'ATP et le NADPH produits pendant la phase "
        "claire pour fixer le dioxyde de carbone et fabriquer du glucose. L'enzyme clé de cette "
        "fixation s'appelle la RuBisCO, l'enzyme la plus abondante sur Terre.",
    ],
    "introduction-python": [
        "Python est un langage de programmation interprété, créé par Guido van Rossum et publié en "
        "1991. Sa syntaxe lisible repose sur l'indentation plutôt que sur des accolades. Une variable "
        "se crée par simple affectation, par exemple age = 30, et son type est déterminé à "
        "l'exécution : on parle de typage dynamique.",
        "Les listes sont des collections ordonnées et modifiables, écrites entre crochets, comme "
        "notes = [12, 15, 9]. Les dictionnaires associent des clés à des valeurs et s'écrivent entre "
        "accolades. Les tuples ressemblent aux listes mais sont immuables : on ne peut pas modifier "
        "leurs éléments après création.",
        "Une fonction se définit avec le mot-clé def, suivi de son nom et de ses paramètres. Le mot-clé "
        "return renvoie une valeur à l'appelant. Pour parcourir une collection, on utilise la boucle "
        "for, par exemple for note in notes. Les conditions s'écrivent avec if, elif et else.",
    ],
}


def escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[str]) -> bytes:
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, text in enumerate(pages):
        lines = textwrap.wrap(text, 85)
        body = "BT /F1 12 Tf 14 TL 50 740 Td " + " T* ".join(f"({escape(l)}) Tj" for l in lines) + " ET"
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objs.append(f"<< /Length {len(body.encode('cp1252'))} >>\nstream\n{body}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body.encode("cp1252") + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "samples"
    target.mkdir(exist_ok=True)
    for name, pages in COURSES.items():
        (target / f"{name}.pdf").write_bytes(build_pdf(pages))
        print("wrote", name)
