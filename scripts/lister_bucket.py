#!/usr/bin/env python3
"""Liste le contenu d'un dossier du bucket public (noms et tailles), sans rien télécharger.

  python3 -I scripts/lister_bucket.py PHerc0139/segments/ [profondeur]
"""
import html
import re
import sys
import urllib.parse
import urllib.request

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"


def lister(prefixe, pages=5):
    """Renvoie (sous-dossiers, [(clé, taille)]) pour un préfixe."""
    jeton, dossiers, fichiers = None, [], []
    for _ in range(pages):
        q = {"list-type": "2", "prefix": prefixe, "delimiter": "/", "max-keys": "1000"}
        if jeton:
            q["continuation-token"] = jeton
        x = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).read().decode()
        dossiers += [html.unescape(p) for p in re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix>", x)]
        fichiers += [(html.unescape(k), int(s)) for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", x)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m:
            break
        jeton = html.unescape(m.group(1))
    return dossiers, fichiers


def arbre(prefixe, profondeur, retrait=""):
    dossiers, fichiers = lister(prefixe)
    for cle, taille in fichiers[:30]:
        print(f"{retrait}{cle[len(prefixe):]}  {taille / 1e6:.2f} Mo")
    if len(fichiers) > 30:
        print(f"{retrait}... +{len(fichiers) - 30} fichiers")
    for d in dossiers[:60]:
        print(f"{retrait}{d[len(prefixe):]}")
        if profondeur > 0 and not d.rstrip("/").endswith(".zarr"):
            arbre(d, profondeur - 1, retrait + "   ")


if __name__ == "__main__":
    arbre(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
