#!/usr/bin/env python3
"""De quels autres rouleaux d'Herculanum le texte de PHerc. Paris 4 est-il le plus proche par son vocabulaire ?

Corpus : les papyrus d'Herculanum édités dans le Digital Corpus of Literary Papyri (externe/idp.data/DCLP).
Chaque texte est réduit à ses mots (sans accents), chaque mot à ses cinq premières lettres ; on pondère
les mots rares (TF-IDF) et on mesure la ressemblance (cosinus) avec Paris 4. Ce n'est pas une preuve
d'identité d'œuvre : cela dit seulement de quels traités connus de la bibliothèque le sujet est le plus voisin.

  .venv/bin/python -I scripts/voisins_bibliotheque.py
"""
import math
import pathlib
import re
import unicodedata
from collections import Counter

RACINE = pathlib.Path(__file__).resolve().parent.parent
DCLP = RACINE / "externe/idp.data/DCLP"
PARIS4 = "989336"
VIDES = set("καιτοντηντηςτουτωντοιςταιςεστιναλλαγαρμενουκουδεουτεωστεπροςπεριαυτοναυτηναυτουαυτωναυτοιςειναιτουτοταυτατουτωντουτουδιοτιεπει".split())


def mots(xml):
    m = re.search(r'<div[^>]*type="edition".*?</body>', xml, flags=re.S)
    if not m:
        return []
    e = m.group(0)
    e = re.sub(r"<(head|note|reg|corr|desc|num|ref|foreign)\b[^>]*>.*?</\1>", " ", e, flags=re.S)
    e = re.sub(r"<gap\b[^>]*?/>|<gap\b[^>]*>.*?</gap>", " # ", e, flags=re.S)
    e = re.sub(r'<lb[^>]*break="no"[^>]*/>', "", e)                       # mot coupé en fin de ligne : on recolle
    e = re.sub(r"<[^>]+>", "", e)
    t = unicodedata.normalize("NFD", e).replace("ͅ", "ι")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().replace("ς", "σ").replace("ϲ", "σ")
    return [w for w in re.findall(r"[α-ω]+", t) if len(w) >= 4]


def main():
    docs = {}
    for f in sorted(DCLP.rglob("*.xml")):
        x = f.read_text(encoding="utf-8", errors="ignore")
        h = re.search(r'<idno type="dclp-hybrid">([^<]*)</idno>', x)
        if not h or not (h.group(1).lower().startswith("p.herc") or f.stem == PARIS4):
            continue
        titre = re.search(r"<title>(.*?)</title>", x, flags=re.S)
        w = mots(x)
        if len(w) >= 150 or f.stem == PARIS4:
            docs[f.stem] = (h.group(1), re.sub(r"\s+", " ", titre.group(1)).strip() if titre else "", w)
    print(f"{len(docs)} papyrus d'Herculanum avec au moins 150 mots dans le corpus ; Paris 4 : {len(docs[PARIS4][2])} mots")
    df = Counter(); tf = {}
    for k, (_, _, w) in docs.items():
        c = Counter(x[:5] for x in w if x not in VIDES); tf[k] = c; df.update(c.keys())
    n = len(docs)
    def vec(c):
        v = {s: (1 + math.log(f)) * math.log(n / df[s]) for s, f in c.items() if df[s] < 0.6 * n}
        z = math.sqrt(sum(a * a for a in v.values())) or 1.0
        return {s: a / z for s, a in v.items()}
    V = {k: vec(c) for k, c in tf.items()}
    q = V[PARIS4]
    print("mots les plus caractéristiques de Paris 4 :", ", ".join(s for s, _ in sorted(q.items(), key=lambda t: -t[1])[:22]))
    res = sorted(((sum(a * V[k].get(s, 0) for s, a in q.items()), k) for k in docs if k != PARIS4), reverse=True)
    print("\nrouleaux d'Herculanum les plus proches par le vocabulaire (1 = identique, 0 = rien de commun) :")
    for sc, k in res[:15]:
        h, t, w = docs[k]
        communs = sorted(((q[s] * V[k][s], s) for s in q if s in V[k]), reverse=True)[:6]
        print("  %.3f  %-22s %-58s (%5d mots)  %s" % (sc, h, t[:58], len(w), " ".join(s for _, s in communs)))
    print("\nressemblance médiane avec les autres rouleaux : %.3f" % sorted(s for s, _ in res)[len(res) // 2])
    # recensement des œuvres de la bibliothèque dans ce corpus
    c = Counter(re.sub(r"\s*\(?\?\)?", "", t.split(" p. ")[0])[:48] for _, (h, t, w) in docs.items())
    print("\ntitres donnés par le corpus aux rouleaux d'Herculanum (les plus fréquents) :")
    for t, k in c.most_common(18):
        print("  %3d  %s" % (k, t))


if __name__ == "__main__":
    main()
