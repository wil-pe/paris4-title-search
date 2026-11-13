#!/usr/bin/env python3
"""Régénère catalogue/volumes.csv à partir des métadonnées officielles du bucket public.

Lit metadata.min.json (mis à jour par l'équipe Vesuvius Challenge), interroge l'en-tête
Zarr de chaque volume pour ses dimensions, puis croise avec catalogue/statut_lecture.csv.

Usage : python3 -I scripts/construire_catalogue.py
Aucune dépendance hors bibliothèque standard.
"""
import concurrent.futures as cf
import csv
import gzip
import json
import pathlib
import urllib.request

RACINE = pathlib.Path(__file__).resolve().parent.parent
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
# Correspondance racine S3 -> URL HTTPS publique (pas besoin de compte AWS).
RACINES_HTTPS = {
    "s3://vesuvius-challenge-open-data": BUCKET,
    "s3://vesuvius-challenge": "https://data.aws.ash2txt.org/",
}


def lire_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        brut = r.read()
    if brut[:2] == b"\x1f\x8b":
        brut = gzip.decompress(brut)
    return json.loads(brut)


def url_https(origine):
    racine = origine["access_roots"][0]["url"].rstrip("/")
    base = RACINES_HTTPS.get(racine, racine + "/")
    return base + origine["path"]


def entete(url):
    """Dimensions du niveau 0 et nombre de niveaux de la pyramide OME-Zarr."""
    try:
        z = lire_json(url + "0/.zarray")
        niveaux = len(lire_json(url + ".zattrs")["multiscales"][0]["datasets"])
        return z["shape"], z["dtype"], niveaux
    except Exception:
        return None, None, None


def main():
    statuts = {}
    with open(RACINE / "catalogue" / "statut_lecture.csv", newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            statuts[ligne["sample_id"]] = ligne

    echantillons = lire_json(BUCKET + "metadata.min.json")["samples"]
    inconnus = sorted(set(echantillons) - set(statuts))
    if inconnus:
        print("ATTENTION, échantillons absents de statut_lecture.csv :", ", ".join(inconnus))

    taches = []
    for sid, s in sorted(echantillons.items()):
        for vid, v in sorted(s["volumes"].items()):
            for d in v["data"]:
                if d["type"] == "ome-zarr":
                    taches.append((sid, vid, s, v, url_https(d["origins"][0])))

    with cf.ThreadPoolExecutor(12) as ex:
        entetes = list(ex.map(lambda t: entete(t[4]), taches))

    lignes = []
    for (sid, vid, s, v, url), (forme, dtype, niveaux) in zip(taches, entetes):
        p = v["properties"]
        scan = s["scans"].get(v.get("scan_id"), {})
        lieu = (scan.get("creation", {}).get("metadata") or {}).get("location", "")
        st = statuts.get(sid, {})
        ligne = {
            "sample_id": sid,
            "type": s["sample"]["properties"].get("type", ""),
            "statut": st.get("statut", "?"),
            "prix_ouverts": st.get("prix_ouverts", ""),
            "volume_id": vid,
            "pixel_um": p.get("pixel_size_um"),
            "energie_keV": p.get("energy_keV"),
            "lieu_scan": lieu,
            "licence": (p.get("license") or {}).get("name", ""),
            "z": "", "y": "", "x": "", "hauteur_mm": "", "diametre_mm": "",
            "niveaux": niveaux or "",
            "brut_L0_Go": "",
            "nb_segments": len(s["segments"]),
            "url": url,
        }
        if forme:
            octets = forme[0] * forme[1] * forme[2] * (2 if "2" in dtype else 1)
            px = p.get("pixel_size_um") or 0
            ligne.update(z=forme[0], y=forme[1], x=forme[2],
                         hauteur_mm=round(forme[0] * px / 1000, 1),
                         diametre_mm=round(max(forme[1], forme[2]) * px / 1000, 1),
                         brut_L0_Go=round(octets / 1e9, 1))
        lignes.append(ligne)

    sortie = RACINE / "catalogue" / "volumes.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    manquants = sum(1 for l in lignes if not l["z"])
    print(f"{len(lignes)} volumes écrits dans {sortie} ({manquants} sans en-tête lisible)")


if __name__ == "__main__":
    main()
