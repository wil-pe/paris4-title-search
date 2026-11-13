#!/usr/bin/env python3
"""Cherche si le texte d'un papyrus recoupe celui d'un autre papyrus littéraire connu.

Corpus : les éditions numériques du Digital Corpus of Literary Papyri (papyri.info), clonées
dans externe/idp.data/DCLP. Chaque texte est réduit à ses lettres grecques (sans accents,
sans ponctuation, sigma unique, iota souscrit développé) ; les lacunes coupent le texte.

Méthode : on repère toutes les suites de K lettres communes entre le papyrus interrogé et
chaque autre texte, puis on cherche des suites alignées (même décalage entre les deux textes,
à quelques dizaines de lettres près). Des mots courants donnent des rencontres dispersées ;
un vrai passage commun donne une chaîne de rencontres alignées.

  .venv/bin/python -I scripts/recherche_texte.py 989336            # PHerc. Paris 4
  .venv/bin/python -I scripts/recherche_texte.py 989336 --k 7 --top 15
"""
import argparse
import pathlib
import pickle
import re
import unicodedata
from collections import defaultdict

RACINE = pathlib.Path(__file__).resolve().parent.parent
DCLP = RACINE / "externe/idp.data/DCLP"
CACHE = RACINE / "data/texte/dclp_normalise.pkl"
BANDE = 40          # tolérance sur le décalage, en lettres (lacunes de longueurs différentes)


def normaliser(texte):
    """Lettres grecques minuscules seulement ; '#' marque une lacune."""
    t = unicodedata.normalize("NFD", texte).replace("ͅ", "ι")
    out = []
    for c in t:
        if c == "#":
            out.append("#")
            continue
        if unicodedata.combining(c):
            continue
        c = c.lower()
        if c in "ςϲ":
            c = "σ"
        if "α" <= c <= "ω":
            out.append(c)
    return re.sub(r"#+", "#", "".join(out))


def lire_edition(chemin):
    """Renvoie (titre, texte normalisé de l'édition grecque) d'un fichier DCLP."""
    x = chemin.read_text(encoding="utf-8", errors="ignore")
    titre = re.search(r"<title>(.*?)</title>", x, flags=re.S)
    m = re.search(r'<div[^>]*type="edition".*?</body>', x, flags=re.S)
    if not m:
        return (titre.group(1).strip() if titre else ""), ""
    e = m.group(0)
    e = re.sub(r"<(head|note|reg|corr|desc|num|ref|foreign)\b[^>]*>.*?</\1>", " ", e, flags=re.S)   # hors texte du papyrus
    e = re.sub(r"<gap\b[^>]*?/>|<gap\b[^>]*>.*?</gap>", "#", e, flags=re.S)
    e = re.sub(r"<[^>]+>", "", e)
    return (titre.group(1).strip() if titre else ""), normaliser(e)


def corpus():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    docs = {}
    for f in sorted(DCLP.rglob("*.xml")):
        titre, texte = lire_edition(f)
        if len(texte.replace("#", "")) >= 40:
            docs[f.stem] = (titre, texte)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(docs))
    return docs


def graines(texte, k):
    """Positions de chaque suite de k lettres ne traversant pas de lacune."""
    d = defaultdict(list)
    for i in range(len(texte) - k + 1):
        g = texte[i:i + k]
        if "#" not in g:
            d[g].append(i)
    return d


def comparer(requete, cible, k):
    """Meilleure chaîne alignée : (nombre de lettres de la requête couvertes, décalage, rencontres)."""
    gq = graines(requete, k)
    hits = []
    for j in range(len(cible) - k + 1):
        g = cible[j:j + k]
        if g in gq:
            hits.extend((i, j) for i in gq[g])
    if not hits:
        return 0, 0, []
    par_diag = defaultdict(list)
    for i, j in hits:
        par_diag[(j - i) // BANDE].append((i, j))
    meilleur = (0, 0, [])
    for d in par_diag:
        h = par_diag[d] + par_diag.get(d + 1, [])
        couvert = set()
        for i, _ in h:
            couvert.update(range(i, i + k))
        if len(couvert) > meilleur[0]:
            meilleur = (len(couvert), d * BANDE, sorted(h))
    return meilleur


def extrait(texte, i, n=60):
    return texte[max(0, i - 10):i + n]


def chercher(ident, k=8, top=12, docs=None, silencieux=False):
    docs = docs or corpus()
    titre, requete = docs[ident]
    resultats = []
    for autre, (t, texte) in docs.items():
        if autre == ident:
            continue
        couvert, decalage, hits = comparer(requete, texte, k)
        if couvert:
            resultats.append((couvert, autre, t, hits))
    resultats.sort(reverse=True)
    if not silencieux:
        print(f"{ident} « {titre} » : {len(requete.replace('#', ''))} lettres, comparé à {len(docs) - 1} textes (suites de {k} lettres)")
        for couvert, autre, t, hits in resultats[:top]:
            i, j = hits[0]
            print(f"  {couvert:4d} lettres alignées | {autre} {t[:70]}")
            print(f"        ici   : …{extrait(requete, i)}…")
            print(f"        autre : …{extrait(docs[autre][1], j)}…")
    return resultats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ident", help="numéro DCLP/TM du papyrus interrogé (989336 = PHerc. Paris 4)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    chercher(a.ident, a.k, a.top)
