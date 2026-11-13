#!/usr/bin/env python3
"""Télécharge un niveau de résolution d'un volume (OME-Zarr) dans data/.

Chaque volume existe en 6 niveaux : 0 = pleine résolution, puis chaque niveau divise
chaque axe par 2 (donc le poids par 8). Le niveau 5 d'un rouleau pèse ~0,1 Go, le
niveau 0 de 0,7 à 123 To.

Par défaut le script ne télécharge RIEN : il affiche ce qui serait récupéré.
Ajouter --executer pour lancer le téléchargement (reprise possible, les blocs déjà
présents sont sautés).

Exemples :
  python3 -I scripts/telecharger_volume.py PHerc0125 --niveau 5
  python3 -I scripts/telecharger_volume.py PHerc0125 --niveau 5 --executer
  python3 -I scripts/telecharger_volume.py PHercParis4 --volume 20230205180739 --niveau 4 --executer
"""
import argparse
import concurrent.futures as cf
import csv
import itertools
import json
import math
import pathlib
import shutil
import sys
import urllib.error
import urllib.request

RACINE = pathlib.Path(__file__).resolve().parent.parent
MARGE_DISQUE_GO = 5  # espace libre à préserver quoi qu'il arrive


def telecharger(url, tentatives=4):
    """Renvoie les octets, ou None si le bloc n'existe pas (zone vide du masque)."""
    for essai in range(tentatives):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if essai == tentatives - 1:
                raise
        except Exception:
            if essai == tentatives - 1:
                raise


def choisir_volume(sample, volume_id):
    with open(RACINE / "catalogue" / "volumes.csv", newline="", encoding="utf-8") as f:
        lignes = [l for l in csv.DictReader(f) if l["sample_id"] == sample]
    if not lignes:
        sys.exit(f"Échantillon inconnu : {sample} (voir catalogue/volumes.csv)")
    if volume_id:
        lignes = [l for l in lignes if l["volume_id"] == volume_id]
        if not lignes:
            sys.exit(f"Volume {volume_id} introuvable pour {sample}")
    elif len(lignes) > 1:
        print(f"{sample} a plusieurs volumes, préciser --volume :")
        for l in lignes:
            print(f"  {l['volume_id']}  {l['pixel_um']} µm  {l['energie_keV']} keV  L0 = {l['brut_L0_Go']} Go")
        sys.exit(1)
    return lignes[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", help="identifiant, ex. PHerc0125")
    ap.add_argument("--volume", help="identifiant du volume si l'échantillon en a plusieurs")
    ap.add_argument("--niveau", type=int, default=5, help="niveau de la pyramide, 0 à 5 (défaut 5)")
    ap.add_argument("--dest", default=str(RACINE / "data"), help="dossier de destination")
    ap.add_argument("--executer", action="store_true", help="lancer réellement le téléchargement")
    ap.add_argument("--fils", type=int, default=16, help="téléchargements simultanés")
    a = ap.parse_args()

    vol = choisir_volume(a.sample, a.volume)
    url = vol["url"]
    zarray_brut = telecharger(f"{url}{a.niveau}/.zarray")
    if zarray_brut is None:
        sys.exit(f"Niveau {a.niveau} absent pour ce volume")
    zarray = json.loads(zarray_brut)
    forme, blocs = zarray["shape"], zarray["chunks"]
    sep = zarray.get("dimension_separator", ".")
    if zarray.get("compressor") is not None:
        print("Note : blocs compressés, la taille réelle sera inférieure à l'estimation.")
    grille = [math.ceil(f / b) for f, b in zip(forme, blocs)]
    nb = grille[0] * grille[1] * grille[2]
    octets_bloc = blocs[0] * blocs[1] * blocs[2] * (2 if "2" in zarray["dtype"] else 1)
    max_go = nb * octets_bloc / 1e9

    dest = pathlib.Path(a.dest) / a.sample / url.rstrip("/").split("/")[-1]
    existant = next(p for p in dest.parents if p.exists())  # la simulation ne crée aucun dossier
    libre_go = shutil.disk_usage(existant).free / 1e9
    px = float(vol["pixel_um"]) * 2 ** a.niveau
    print(f"{a.sample} / volume {vol['volume_id']} / niveau {a.niveau}")
    print(f"  dimensions (z,y,x) : {forme}   voxel : {px:.1f} µm")
    print(f"  blocs : {nb} x {octets_bloc / 1e6:.1f} Mo  ->  {max_go:.2f} Go au maximum (les blocs vides ne sont pas stockés)")
    print(f"  destination : {dest}")
    print(f"  espace libre : {libre_go:.1f} Go")
    if not a.executer:
        print("Simulation seulement. Ajouter --executer pour télécharger.")
        return
    if max_go > libre_go - MARGE_DISQUE_GO:
        sys.exit(f"Refus : il faut {max_go:.1f} Go et garder {MARGE_DISQUE_GO} Go de marge, "
                 f"seulement {libre_go:.1f} Go libres. Choisir un niveau plus élevé ou un autre disque (--dest).")

    # Métadonnées, pour que le dossier local reste un Zarr valide.
    for nom in (".zgroup", ".zattrs"):
        contenu = telecharger(url + nom)
        if contenu is not None:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / nom).write_bytes(contenu)
    dossier = dest / str(a.niveau)
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / ".zarray").write_bytes(zarray_brut)

    def un_bloc(idx):
        cle = sep.join(map(str, idx))
        cible = dossier / cle
        if cible.exists() and cible.stat().st_size > 0:
            return 0
        contenu = telecharger(f"{url}{a.niveau}/{cle}")
        if contenu is None:
            return -1
        cible.parent.mkdir(parents=True, exist_ok=True)
        tmp = cible.with_name(cible.name + ".part")
        tmp.write_bytes(contenu)
        tmp.rename(cible)
        return len(contenu)

    fait = vides = octets = 0
    with cf.ThreadPoolExecutor(a.fils) as ex:
        for n in ex.map(un_bloc, itertools.product(*(range(g) for g in grille))):
            fait += 1
            if n < 0:
                vides += 1
            else:
                octets += n
            if fait % 50 == 0 or fait == nb:
                print(f"\r  {fait}/{nb} blocs, {octets / 1e9:.2f} Go téléchargés, {vides} vides", end="", flush=True)
    print(f"\nTerminé : {dossier}")


if __name__ == "__main__":
    main()
