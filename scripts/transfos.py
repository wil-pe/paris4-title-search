#!/usr/bin/env python3
"""Reconstruit les matrices de passage entre les scans de PHerc. Paris 4 à partir des métadonnées publiques.

Le fichier public metadata.min.json donne, pour chaque échantillon, des matrices affines 3 x 4 entre ses
volumes (samples.<id>.sample.properties.volume_transforms). Elles agissent sur des coordonnées (x, y, z)
en voxels pleine résolution du volume de départ et rendent des voxels pleine résolution du volume d'arrivée.
Les autres scripts les attendent en 4 x 4 dans data/PHercParis4/ :

  transfo_24_vers_1129.npy   scan 2,4 µm 78 keV (20260411134726) -> scan 1,129 µm (20260608103018)
  transfo_78_vers_137.npy    scan 2,4 µm 78 keV (20260411134726) -> scan 2,4 µm 137 keV (20260323153942)

  .venv/bin/python -I scripts/transfos.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import construire_catalogue as CC  # noqa: E402
import volume as V  # noqa: E402

ECHANTILLON = "PHercParis4"
PAIRES = {
    "transfo_24_vers_1129.npy": ("20260411134726", "20260608103018"),
    "transfo_78_vers_137.npy": ("20260411134726", "20260323153942"),
}


def matrice(transfos, depart, arrivee):
    for bloc in transfos:
        if bloc["from_volume_id"] == depart:
            for t in bloc["transforms"]:
                if t["to_volume_id"] == arrivee:
                    return np.vstack([np.asarray(t["matrix"], np.float64), [0, 0, 0, 1]])
    raise SystemExit(f"aucune matrice {depart} -> {arrivee} dans les métadonnées")


if __name__ == "__main__":
    meta = CC.lire_json(CC.BUCKET + "metadata.min.json")["samples"][ECHANTILLON]
    transfos = meta["sample"]["properties"]["volume_transforms"]
    dossier = V.DATA / ECHANTILLON
    dossier.mkdir(parents=True, exist_ok=True)
    for nom, (a, b) in PAIRES.items():
        M = matrice(transfos, a, b)
        np.save(dossier / nom, M)
        echelle = np.linalg.norm(M[:3, :3], axis=0).mean()
        print(f"{nom} : {a} -> {b}, facteur d'échelle {echelle:.4f}, écrit dans {dossier / nom}")
