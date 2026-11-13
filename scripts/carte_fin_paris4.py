#!/usr/bin/env python3
"""Carte d'encre de la fin de PHerc. Paris 4, découpée en tranches de profondeur.

Projette une prédiction d'encre 3D sur le segment des spires internes (w010-027) à 9,6 µm/px,
de x = 72 000 à 170 000 px (172 à 408 mm) sur toute la hauteur. Pour chaque pixel on garde le
maximum le long de la normale, sa profondeur, et le maximum dans six tranches de profondeur :
les tranches proches de t = 0 portent l'encre de la feuille elle-même, les tranches t > 0
celle de la spire voisine extérieure (les « fantômes »).

Le calcul se fait par bandes de hauteur, enregistrées une à une : relancer la commande reprend
là où elle s'est arrêtée. Les blocs téléchargés restent en cache dans data/ (~6 Go par modèle).

  .venv/bin/python -I scripts/carte_fin_paris4.py --modele student            # tout
  .venv/bin/python -I scripts/carte_fin_paris4.py --modele avril --part 0/2   # une bande sur deux
  .venv/bin/python -I scripts/carte_fin_paris4.py --modele student --assembler
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rendu_surface as R  # noqa: E402
import volume as V  # noqa: E402

MAILLAGE = V.DATA / "PHercParis4/segments/20260701183124-w010-027/mesh/20260701183124-on-20260411134726-2.4um.tifxyz"
PRED = "PHercParis4/representations/predictions/ink-3d/"
# modèle -> (chemin du Zarr, niveau à 9,6 µm)
MODELES = {
    "student": (PRED + "20260411134726-ink3d-20260926084000-student-v12A-25000-c7766be9-L1.zarr/", 1),
    "avril": (PRED + "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/", 2),
}
Y0, Y1, X0, X1 = 0, 69760, 72000, 170000      # pixels pleine résolution (2,4 µm) du segment
PAS = 4                                        # 9,6 µm par pixel de sortie
T = 14                                         # ±134 µm le long de la normale
TRANCHES = [(-14, -9), (-8, -5), (-4, -1), (0, 3), (4, 7), (8, 14)]
HAUTEUR = 3400                                 # hauteur d'une bande, en pixels pleine résolution
LARGEUR = 14000                                # largeur d'une fenêtre : plus d'un tour de spire
SORTIE = V.DATA / "PHercParis4/carte"


def bandes():
    return [(y, min(y + HAUTEUR, Y1)) for y in range(Y0, Y1, HAUTEUR)]


def fenetres():
    return [(x, min(x + LARGEUR, X1)) for x in range(X0, X1, LARGEUR)]


def calculer(modele, part):
    chemin, niveau = MODELES[modele]
    dossier = SORTIE / modele
    dossier.mkdir(parents=True, exist_ok=True)
    i, n = (int(a) for a in part.split("/"))
    for j, (ya, yb) in enumerate(bandes()):
        if j % n != i:
            continue
        for xa, xb in fenetres():
            cible = dossier / f"bande_{ya:05d}_x{xa:06d}.npz"
            if cible.exists():
                continue
            # toutes les fenêtres d'une bande relisent les mêmes blocs : le cache disque évite de les retélécharger
            img, prof, tr = R.rendre(chemin, niveau, 4, MAILLAGE, (ya, yb, xa, xb), pas=PAS, T=T, bande=10 ** 9, tranches=TRANCHES)
            tmp = dossier / f"bande_{ya:05d}_x{xa:06d}.part.npz"
            np.savez_compressed(tmp, img=img.astype(np.uint8), prof=prof.astype(np.int8), **{f"s{k}": a for k, a in enumerate(tr)})
            tmp.rename(cible)
        print(f"{modele} : bande {ya}-{yb} enregistrée", flush=True)


def assembler(modele):
    """Réunit les morceaux en cinq images : maximum global, profondeur, et trois regroupements de tranches."""
    dossier = SORTIE / modele
    groupes = {"img": ["img"], "prof": ["prof"], "dedans": ["s0"], "propre": ["s1", "s2", "s3"], "dehors": ["s4", "s5"]}
    lignes = {g: [] for g in groupes}
    for ya, _ in bandes():
        morceaux = {g: [] for g in groupes}
        for xa, _ in fenetres():
            with np.load(dossier / f"bande_{ya:05d}_x{xa:06d}.npz") as p:
                for g, cles in groupes.items():
                    morceaux[g].append(np.max([p[c] for c in cles], axis=0) if len(cles) > 1 else p[cles[0]])
        for g in groupes:
            lignes[g].append(np.concatenate(morceaux[g], axis=1))
    for g in groupes:
        a = np.concatenate(lignes[g], axis=0)
        np.save(SORTIE / f"{modele}_{g}.npy", a)
        print(modele, g, a.shape, a.dtype, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modele", choices=list(MODELES), required=True)
    ap.add_argument("--part", default="0/1", help="i/n : ne traiter que les bandes d'indice i modulo n")
    ap.add_argument("--assembler", action="store_true", help="réunir les bandes en images complètes")
    a = ap.parse_args()
    if a.assembler:
        assembler(a.modele)
    else:
        calculer(a.modele, a.part)
