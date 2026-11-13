#!/usr/bin/env python3
"""Lecture « à l'aveugle » des pages de la visionneuse façon livre, par des sous-tâches (agents de lecture).

Chaque page de apercus/livre/ est donnée à un agent sous forme de planches : une vue d'ensemble réduite et
des bandes horizontales de 60 mm agrandies, lissées (le lissage joue le rôle du « dézoom » : les lettres
ressortent du bruit), bordées de règles en millimètres dans le repère de la page. L'agent rend ses lectures
ligne par ligne ; elles sont rangées dans apercus/livre/lectures/pNNN.json et affichées par la visionneuse
(visionneuse_livre.py --texte).

Convention de lecture : capitales grecques sans accents ; MAJUSCULE = lettre sûre, minuscule = lettre
incertaine, « . » = lettre présente mais illisible, « [...] » = lacune. Aucune restitution.

  .venv/bin/python -I scripts/lecture_livre.py planches DOSSIER 75 139 151     # planches de lecture
  .venv/bin/python -I scripts/lecture_livre.py noter 151 153                   # note une lecture face à l'édition
  .venv/bin/python -I scripts/lecture_livre.py importer DOSSIER [journal.jsonl]  # valide et range les lectures rendues
  .venv/bin/python -I scripts/lecture_livre.py bilan                           # ce que les lectures ont donné, page par page

« noter » sert aux colonnes témoins : une colonne déjà éditée, lue sans que l'agent connaisse l'édition, donne
la fiabilité réelle de la méthode (part des lettres de l'édition retrouvées, part des lettres lues qui sont justes).
"""
import difflib
import json
import pathlib
import re
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import visionneuse_livre as L  # noqa: E402

LECTURES = L.DOS / "lectures"
ECH = 20.8                 # pixels par mm des bandes (1,6 fois la carte du rouleau entier)
HAUT, PAS = 60.0, 52.0     # hauteur d'une bande et pas entre deux bandes, en mm
LISSAGE = 0.115            # écart-type du lissage, en mm (1,5 pixel de la carte du rouleau entier)
GAIN = {"k": 1.6, "h": 2.5, "c": 1.0, "s": 1.0, "a": 1.0}


def donnees():
    t = (L.DOS / "donnees.js").read_text(encoding="utf-8")
    return json.loads(t[len("const D = "):].rstrip().rstrip(";"))


def police(n):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def avec_regles(img, x0, y0, ech):
    """Colle l'image sous une règle des x et à droite d'une règle des y (mm, repère de la page)."""
    mg, mh = 64, 34
    out = Image.new("L", (img.width + mg, img.height + mh), 26); out.paste(img, (mg, mh))
    d = ImageDraw.Draw(out); f = police(17)
    for m in range(int(np.ceil(x0)), int(x0 + img.width / ech) + 1):
        X = mg + (m - x0) * ech; g = m % 10 == 0
        d.line([(X, mh - (14 if g else 6 if m % 5 == 0 else 3)), (X, mh)], fill=235)
        if g and X < out.width - 50:
            d.text((X + 3, 2), str(m), fill=235, font=f)
    for m in range(int(np.ceil(y0)), int(y0 + img.height / ech) + 1):
        Y = mh + (m - y0) * ech; g = m % 10 == 0
        d.line([(mg - (14 if g else 6 if m % 5 == 0 else 3), Y), (mg, Y)], fill=235)
        if g and Y < out.height - 12:
            d.text((4, Y - 9), f"y {m}", fill=235, font=f)
    return out


def planches(num, dossier, D=None):
    """Écrit dans dossier/pNNN/ : ensemble.jpg, puis k_1.jpg, h_1.jpg… (bandes de haut en bas). Renvoie la liste."""
    p = (D or donnees())["pages"][num - 1]
    dos = pathlib.Path(dossier) / f"p{num:03d}"; dos.mkdir(parents=True, exist_ok=True)
    fichiers = []
    for c in [c for c in ("c", "s", "k", "h") if c in p["cartes"]][:2] if p["repere"] == "segment" else ("k", "h"):
        a = np.asarray(Image.open(L.DOS / "pages" / f"p{num:03d}_{c}.jpg").convert("L")).astype(np.float32)
        px = a.shape[1] / p["w"]                              # pixels par mm de cette carte
        b = np.clip(gaussian_filter(a, LISSAGE * px) * GAIN[c], 0, 255).astype(np.uint8)
        if not fichiers:                                      # vue d'ensemble : la première carte, à 7 px/mm
            e = Image.fromarray(b).resize((int(p["w"] * 7), int(p["h"] * 7)), Image.LANCZOS)
            avec_regles(e, p["x0"], p["y0"], 7).save(dos / "ensemble.jpg", quality=88); fichiers.append("ensemble.jpg")
        y, i = 0.0, 1
        while y < p["h"] - 8:
            r0, r1 = int(y * px), min(a.shape[0], int((y + HAUT) * px))
            t = Image.fromarray(b[r0:r1]).resize((int(p["w"] * ECH), int((r1 - r0) / px * ECH)), Image.LANCZOS)
            avec_regles(t, p["x0"], p["y0"] + y, ECH).save(dos / f"{c}_{i}.jpg", quality=90); fichiers.append(f"{c}_{i}.jpg")
            y += PAS; i += 1
    return dos, fichiers


# ------------------------------------------------------------------ notation face à l'édition

def lettres(t, garder_casse=False):
    """Lettres grecques seulement, sans accents ; sigma final et lunaire ramenés à sigma."""
    t = unicodedata.normalize("NFD", t).replace("ͅ", "ι")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("Ϲ", "Σ").replace("ϲ", "σ").replace("ς", "σ").replace("Є", "Ε").replace("є", "ε")
    t = "".join(c for c in t if "α" <= c.lower() <= "ω")
    return t if garder_casse else t.upper()


def noter(num, D=None):
    """Aligne la lecture de la page sur le texte de l'édition (lettres visibles : sans les restitutions entre crochets)."""
    D = D or donnees(); p = D["pages"][num - 1]
    lec = json.loads((LECTURES / f"p{num:03d}.json").read_text(encoding="utf-8"))
    ed = lettres("".join(re.sub(r"\[[^\]]*\]|\([^)]*\)|– ca\. \d+ –", "", l["t"]) for l in D["edition"][str(p["ed"])]))
    lu = lettres("".join(l["lu"] for l in sorted(lec["lignes"], key=lambda l: l["y"])), garder_casse=True)
    blocs = difflib.SequenceMatcher(None, ed, lu.upper(), autojunk=False).get_matching_blocks()
    juste = np.zeros(len(lu), bool)
    for b in blocs:
        juste[b.b:b.b + b.size] = True
    sur = np.array([c.isupper() for c in lu])
    r = dict(page=num, colonne=p["ed"], lettres_edition=len(ed), lettres_lues=len(lu),
             retrouvees=round(float(juste.sum()) / max(len(ed), 1), 3),
             justes_parmi_sures=round(float(juste[sur].mean()), 3) if sur.any() else None, nb_sures=int(sur.sum()),
             justes_parmi_incertaines=round(float(juste[~sur].mean()), 3) if (~sur).any() else None, nb_incertaines=int((~sur).sum()))
    return r


# ------------------------------------------------------------------ rangement des lectures rendues par les sous-tâches

LATIN = str.maketrans("aAoOeEiItTnNkKpPxXyYbBhHmMzZvuc", "αΑοΟεΕιΙτΤνΝκΚρΡχΧυΥβΒηΗμΜζΖνυσ")   # sosies latins tapés par mégarde


def nettoyer(d, p):
    """Vérifie une lecture et la remet en forme ; lève ValueError si elle est inutilisable."""
    if not isinstance(d, dict) or not isinstance(d.get("lignes"), list) or not isinstance(d.get("zones"), list):
        raise ValueError("structure")
    lignes = []
    for l in d["lignes"]:
        y, x0, x1 = float(l["y"]), float(l["x0"]), float(l["x1"])
        if not (p["y0"] - 5 <= y <= p["y0"] + p["h"] + 5 and p["x0"] - 5 <= x0 <= x1 <= p["x0"] + p["w"] + 5):
            raise ValueError(f"ligne hors de la page : y {y}, x {x0}–{x1}")
        lignes.append(dict(y=round(y, 1), x0=round(x0, 1), x1=round(x1, 1), lu="".join(str(l["lu"]).split()).translate(LATIN),
                           confiance=max(0, min(3, int(l["confiance"])))))
    return dict(page=p["num"], zones=d["zones"], entrecolonnes_x=[float(v) for v in d.get("entrecolonnes_x") or []], lignes=lignes,
                mots=[dict(y=float(m["y"]), mot=str(m["mot"]), sur=bool(m["sur"])) for m in d.get("mots") or [] if str(m.get("mot", "")).strip()],
                remarques=str(d.get("remarques", "")))


def trouver(o):
    """Toutes les lectures (dictionnaires avec page, lignes et zones) enfouies dans un objet JSON quelconque."""
    if isinstance(o, dict):
        if {"page", "lignes", "zones"} <= o.keys():
            yield o
        for v in o.values():
            yield from trouver(v)
    elif isinstance(o, list):
        for v in o:
            yield from trouver(v)
    elif isinstance(o, str) and '"lignes"' in o and o.lstrip().startswith("{"):
        try:
            yield from trouver(json.loads(o))
        except ValueError:
            pass


def importer(dossier, journal=None):
    """Range dans apercus/livre/lectures/ les fichiers pNNN.json écrits par les sous-tâches (et, à défaut, les copies
    validées du journal du workflow). Ne remplace jamais une lecture déjà rangée."""
    D = donnees(); LECTURES.mkdir(exist_ok=True)
    sources = {}
    for f in sorted(pathlib.Path(dossier).glob("p[0-9]*.json")):
        try:
            sources[int(f.stem[1:])] = json.loads(re.sub(r"^```(?:json)?|```$", "", f.read_text(encoding="utf-8").strip()).strip())
        except ValueError as e:
            print(f"  {f.name} : JSON invalide ({e})")
    if journal:
        for ligne in pathlib.Path(journal).read_text(encoding="utf-8").splitlines():
            try:
                for d in trouver(json.loads(ligne)):
                    sources.setdefault(int(d["page"]), d)
            except (ValueError, TypeError):
                pass
    faits, rejets = [], []
    for n, d in sorted(sources.items()):
        cible = LECTURES / f"p{n:03d}.json"
        if cible.exists() or not 1 <= n <= len(D["pages"]):
            continue
        try:
            cible.write_text(json.dumps(nettoyer(d, D["pages"][n - 1]), ensure_ascii=False, indent=1), encoding="utf-8"); faits.append(n)
        except (ValueError, KeyError, TypeError) as e:
            rejets.append((n, str(e)))
    attendues = [p["num"] for p in D["pages"] if p["type"] != "vierge" and not p.get("ed")]
    manquent = [n for n in attendues if not (LECTURES / f"p{n:03d}.json").exists()]
    print(f"{len(faits)} lectures rangées ; rejetées : {rejets} ; il manque {len(manquent)} pages sans édition : {manquent}")
    return faits, rejets, manquent


def bilan():
    """Ce que les lectures ont donné : par page, lignes repérées, lettres sûres, mots sûrs ; écrit lectures/bilan.json."""
    D = donnees(); lignes_ = []
    for f in sorted(LECTURES.glob("p[0-9]*.json")):
        d = json.loads(f.read_text(encoding="utf-8")); p = D["pages"][d["page"] - 1]
        lu = "".join(l["lu"] for l in d["lignes"])
        lignes_.append(dict(page=d["page"], x=round(p["xr"][0]), temoin=bool(p.get("ed")), lignes=len(d["lignes"]),
                            lues=sum(l["confiance"] >= 2 for l in d["lignes"]), sures=sum(c.isupper() for c in lu),
                            incertaines=sum(c.islower() for c in lu), mots=[m["mot"] for m in d["mots"] if m["sur"]],
                            hypotheses=[m["mot"] for m in d["mots"] if not m["sur"]]))
    (LECTURES / "bilan.json").write_text(json.dumps(lignes_, ensure_ascii=False, indent=1), encoding="utf-8")
    return lignes_


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "planches":
        D = donnees()
        for n in sys.argv[3:]:
            print(*planches(int(n), sys.argv[2], D))
    elif len(sys.argv) > 2 and sys.argv[1] == "noter":
        D = donnees(); notes = [noter(int(n), D) for n in sys.argv[2:]]
        for r in notes:
            print(r)
        (LECTURES / "temoins.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    elif len(sys.argv) > 2 and sys.argv[1] == "importer":
        importer(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif len(sys.argv) > 1 and sys.argv[1] == "bilan":
        b = [x for x in bilan() if not x["temoin"]]
        print(len(b), "pages lues hors témoins ;", sum(x["lignes"] for x in b), "lignes repérées dont", sum(x["lues"] for x in b),
              "lues (confiance ≥ 2) ;", sum(x["sures"] for x in b), "lettres sûres,", sum(x["incertaines"] for x in b), "incertaines")
        print("pages avec mots sûrs :", {x["page"]: x["mots"] for x in b if x["mots"]})
        print("pages les plus lisibles (lettres sûres) :", [(x["page"], x["sures"]) for x in sorted(b, key=lambda x: -x["sures"])[:25]])
    else:
        print(__doc__)
