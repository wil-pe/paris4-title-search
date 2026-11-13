#!/usr/bin/env python3
"""Construit la carte d'encre « sans fantômes » de la fin de PHerc. Paris 4 et liste les candidats.

Entrées : les images par tranche de profondeur produites par carte_fin_paris4.py (--assembler)
et la carte des vis-à-vis produite par voisinage.py.

Pour chaque modèle :
  propre   = maximum des tranches t = -8..+3 (encre posée sur le recto de la feuille elle-même)
  fantome  = encre « propre » de la spire voisine, ramenée à la position qui lui fait face
La carte finale garde l'encre « propre » vue par les deux modèles et met à zéro ce qu'un
vis-à-vis explique. Les candidats sont les amas restants dans la zone sans texte.

  .venv/bin/python -I scripts/carte_sans_fantomes.py
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402

D = C.SORTIE
PAS = C.PAS
UM = 2.4 * PAS                      # 9,6 µm par pixel




def vis_a_vis(image, cote, bloc=1000):
    """Échantillonne `image` à la position du vis-à-vis (cote = 'ext' ou 'int') de chaque pixel."""
    vo = np.load(D / "voisinage.npz")
    pas_g = float(vo["pas_px"])
    u, v = vo[f"u_{cote}"], vo[f"v_{cote}"]
    ok = np.isfinite(u)
    # on bouche les trous par le plus proche voisin pour pouvoir interpoler, et on garde un masque
    idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    u, v = u[tuple(idx)], v[tuple(idx)]
    h, w = image.shape
    sortie = np.zeros((h, w), np.uint8); valide = np.zeros((h, w), bool)
    img = np.asarray(image, dtype=np.float32)
    for i0 in range(0, h, bloc):
        ii, jj = np.meshgrid(np.arange(i0, min(i0 + bloc, h)), np.arange(w), indexing="ij")
        cg = [(ii * PAS + C.Y0) / pas_g, (jj * PAS + C.X0) / pas_g]
        uu = ndi.map_coordinates(u, cg, order=1, mode="nearest"); vv = ndi.map_coordinates(v, cg, order=1, mode="nearest")
        m = ndi.map_coordinates(ok.astype(np.float32), cg, order=1, mode="nearest") > 0.99
        s = ndi.map_coordinates(img, [(vv - C.Y0) / PAS, (uu - C.X0) / PAS], order=1, mode="constant")
        sortie[i0:i0 + bloc] = np.clip(s, 0, 255).astype(np.uint8); valide[i0:i0 + bloc] = m
    return sortie, valide


def zone_sans_texte(forme):
    """Masque de la zone où aucun texte n'est attendu : à droite de l'avant-dernière colonne,
    en excluant le bloc des cinq lignes de la dernière colonne (et son paragraphos)."""
    h, w = forme
    z = np.zeros(forme, bool)
    z[:, (84000 - C.X0) // PAS:] = True
    z[:(17500 - C.Y0) // PAS, (88500 - C.X0) // PAS:(118500 - C.X0) // PAS] = False
    return z


def candidats(residu, risque, bloc=128, seuil_z=6.0, n_max=40):
    """Amas d'encre résiduelle : densité par blocs de 1,2 mm comparée au fond local."""
    h, w = residu.shape
    hb, wb = h // bloc, w // bloc
    zone = zone_sans_texte(residu.shape)
    def par_bloc(a):
        return a[:hb * bloc, :wb * bloc].reshape(hb, bloc, wb, bloc).mean((1, 3))
    dens = par_bloc((residu > 128).astype(np.float32))
    couv = par_bloc(zone.astype(np.float32)) > 0.99
    valide = par_bloc((np.load(D / "student_img.npy", mmap_mode="r") > 0).astype(np.float32)) > 0.2   # blocs où le maillage existe
    ok = couv & valide
    # fond local : médiane et écart robuste sur un voisinage de 25 x 25 blocs (30 mm)
    d = np.where(ok, dens, np.nan)
    med = ndi.generic_filter(d, np.nanmedian, size=25, mode="nearest")
    mad = ndi.generic_filter(np.abs(d - med), np.nanmedian, size=25, mode="nearest") * 1.4826
    z = (d - med) / np.maximum(mad, 0.004)
    chaud = ok & (z > seuil_z)
    lab, n = ndi.label(chaud, structure=np.ones((3, 3)))
    lignes = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        ys, xs = sl
        y0, y1, x0, x1 = ys.start * bloc, ys.stop * bloc, xs.start * bloc, xs.stop * bloc
        r = residu[y0:y1, x0:x1] > 128
        lignes.append(dict(
            score=float((z[sl] * m).sum()), zmax=float(z[sl][m].max()), blocs=int(m.sum()),
            x_mm=(C.X0 + (x0 + x1) / 2 * PAS) * 0.0024, y_mm=(C.Y0 + (y0 + y1) / 2 * PAS) * 0.0024,
            larg_mm=(x1 - x0) * UM / 1000, haut_mm=(y1 - y0) * UM / 1000, densite=float(r.mean()),
            risque=float(risque[y0:y1, x0:x1].mean()), boite=(y0, y1, x0, x1)))
    lignes.sort(key=lambda c: -c["score"])
    return lignes[:n_max], dens, z, ok


def main():
    modeles = [m for m in C.MODELES if (D / f"{m}_propre.npy").exists()]
    print("modèles disponibles :", modeles)
    propres = {m: np.load(D / f"{m}_propre.npy") for m in modeles}
    accord = np.sqrt(np.prod([propres[m].astype(np.float32) for m in modeles], axis=0)) if len(modeles) > 1 else propres[modeles[0]].astype(np.float32)
    accord = np.clip(accord, 0, 255).astype(np.uint8)
    np.save(D / "propre_accord.npy", accord)
    fant = {}
    for cote in ("ext", "int"):
        f, val = vis_a_vis(accord, cote)
        fant[cote] = f; np.save(D / f"fantome_{cote}.npy", f)
        print(f"vis-à-vis {cote} : couvre {100 * val.mean():.0f} % de la carte")
    explique = ndi.binary_dilation(np.maximum(fant["ext"], fant["int"]) > 128, iterations=8)   # ~77 µm de marge
    residu = np.where(explique, 0, accord).astype(np.uint8)
    np.save(D / "residu.npy", residu); np.save(D / "explique.npy", explique)
    zone = zone_sans_texte(accord.shape)
    brut = np.sqrt(np.prod([np.load(D / f"{m}_img.npy").astype(np.float32) for m in modeles], axis=0)) if len(modeles) > 1 else np.load(D / f"{modeles[0]}_img.npy")
    print("zone sans texte, fraction de pixels > 128 : projection épaisse %.4f | encre propre %.4f | après retrait des fantômes %.4f | part expliquée par un vis-à-vis %.4f"
          % ((brut[zone] > 128).mean(), (accord[zone] > 128).mean(), (residu[zone] > 128).mean(), explique[zone].mean()))
    cands, dens, z, ok = candidats(residu, explique)
    np.save(D / "candidats_z.npy", z.astype(np.float32))
    import json
    (D / "candidats.json").write_text(json.dumps(cands, indent=1))
    print("blocs analysés : %d ; blocs au-dessus du seuil : %d ; amas : %d" % (ok.sum(), (ok & (z > 6)).sum(), len(cands)))
    print("  #  score  zmax blocs   x mm   y mm  larg x haut mm  densité  risque fantôme")
    for i, c in enumerate(cands, 1):
        print("%3d %6.0f %5.1f %5d %6.1f %6.1f  %4.1f x %4.1f   %6.3f   %5.2f" % (i, c["score"], c["zmax"], c["blocs"], c["x_mm"], c["y_mm"], c["larg_mm"], c["haut_mm"], c["densite"], c["risque"]))


if __name__ == "__main__":
    main()
