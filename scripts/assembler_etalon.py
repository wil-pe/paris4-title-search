#!/usr/bin/env python3
"""Assemble les tuiles d'étalonnage (texte de PHerc. Paris 4 à l'encre affaiblie) en une seule bande.

`balayage.py etalon` écrit une tuile de 3,1 mm par fichier ; les deux détecteurs d'écriture pâle
(detecteur_pale.py, detecteur_mosaique.py) et comparaison_titre.py attendent la ligne entière dans
data/PHercParis4/balayage/etalon_texte_mosaique.npz : les tuiles mises bout à bout de gauche à droite.

  .venv/bin/python -I scripts/balayage.py etalon etalon_texte 29361 30641 73600 81280 0.6 0.4 0.25 0.15 0
  .venv/bin/python -I scripts/assembler_etalon.py [etalon_texte]
"""
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

B = V.DATA / "PHercParis4/balayage"

if __name__ == "__main__":
    nom = sys.argv[1] if len(sys.argv) > 1 else "etalon_texte"
    fichiers = sorted(glob.glob(str(B / nom / "*.npz")))                      # noms Y_X : l'ordre alphabétique est l'ordre des x
    if not fichiers:
        raise SystemExit(f"aucune tuile dans {B / nom} : lancer d'abord balayage.py etalon")
    tuiles = [np.load(f) for f in fichiers]
    cles = [k for k in tuiles[0].files if not k.endswith("_prof") and all(k in t.files for t in tuiles)]
    mos = {k: np.concatenate([t[k] for t in tuiles], axis=1) for k in cles}
    mos = {k: (v if v.dtype == bool else v.astype(np.float32)) for k, v in mos.items()}
    np.savez_compressed(B / f"{nom}_mosaique.npz", **mos)
    h, w = mos["ok"].shape
    print(f"{len(tuiles)} tuiles -> {B / (nom + '_mosaique.npz')} : {h} x {w} px ({h * 0.0096:.1f} x {w * 0.0096:.1f} mm), {len(cles)} couches")
