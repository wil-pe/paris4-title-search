#!/usr/bin/env python3
"""Cherche une écriture très pâle dans la zone vierge de la fin de PHerc. Paris 4.

Les cartes d'encre publiées sont des scores entre 0 et 255. Une encre trop pâle pour passer le
seuil laisse quand même des scores un peu plus élevés que le papyrus nu. On repasse donc en
« logit » (l'échelle interne du modèle), on moyenne les deux modèles 3D, on retire le fond
lent, puis on cherche des rectangles de la taille d'une ligne d'écriture dont la moyenne
dépasse nettement celle du reste de la zone vierge.

Sont exclus : le texte, les bords du maillage, et tout endroit dont la spire voisine porte du
texte (fantômes). Sortie : data/PHercParis4/carte/candidats_faibles.json et des cartes réduites.

  .venv/bin/python -I scripts/recherche_faible.py
"""
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import rendu_surface as R  # noqa: E402

D = C.SORTIE
F = 4                                   # réduction : 38,4 µm par pixel
MM = 0.0024 * C.PAS * F
HAUTEURS = (2.0, 3.0, 4.5)              # mm : hauteur d'une ligne d'écriture
LARGEURS = (6.0, 12.0, 24.0, 40.0)      # mm : longueur d'un mot ou d'une ligne
BAS, HAUT = -4.0, 0.0                   # bornes de l'évidence d'encre, en logits (score 5 à 128 sur 255)


def logit_reduit(nom):
    """Carte de scores 0..255 -> logits moyennés par blocs de F x F pixels."""
    a = np.load(D / nom, mmap_mode="r")
    v = np.arange(256, dtype=np.float32)
    lut = np.log((v + 0.5) / (255.5 - v)).astype(np.float32)
    h, w = a.shape[0] // F, a.shape[1] // F
    out = np.empty((h, w), np.float32)
    for r in range(0, h, 500):
        bloc = lut[np.asarray(a[r * F:(r + 500) * F, :w * F])]
        out[r:r + 500] = bloc.reshape(-1, F, w, F).mean((1, 3))
    return out


def xmm(col):
    return (C.X0 + col * F * C.PAS) * 0.0024


def col_de(x_mm):
    return int((x_mm / 0.0024 - C.X0) / (F * C.PAS))


def masques(forme):
    """(valide, texte, risque de fantôme) à la résolution réduite."""
    h, w = forme
    (gx, _, gz), k = R._maillage(C.MAILLAGE)                                  # validité lue sur le maillage lui-même
    ok_grille = (gz > 0) & (gx > 0)
    lig = np.clip(((C.Y0 + np.arange(h) * F * C.PAS) / k).astype(int), 0, ok_grille.shape[0] - 1)
    col = np.clip(((C.X0 + np.arange(w) * F * C.PAS) / k).astype(int), 0, ok_grille.shape[1] - 1)
    valide = ok_grille[np.ix_(lig, col)]
    valide &= ((C.X0 + np.arange(w) * F * C.PAS) / k < ok_grille.shape[1])[None, :]
    valide = ndi.binary_erosion(valide, iterations=int(1.5 / MM))            # 1,5 mm des bords et des trous
    texte = np.zeros(forme, bool)
    texte[:, :col_de(199)] = True                                             # colonnes précédentes
    texte[:int(44 / MM), col_de(210):col_de(287)] = True                      # dernière colonne
    # fantômes : le vis-à-vis (de chaque côté) tombe dans une zone de texte
    vo = np.load(D / "voisinage.npz"); pas_g = float(vo["pas_px"])
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cg = [(ii * F * C.PAS + C.Y0) / pas_g, (jj * F * C.PAS + C.X0) / pas_g]
    risque = np.zeros(forme, bool)
    for cote in ("ext", "int"):
        u, v = vo[f"u_{cote}"], vo[f"v_{cote}"]
        ok = np.isfinite(u)
        idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        uu = ndi.map_coordinates(u[tuple(idx)], cg, order=1, mode="nearest")
        vv = ndi.map_coordinates(v[tuple(idx)], cg, order=1, mode="nearest")
        r2 = np.clip(((vv - C.Y0) / (F * C.PAS)).astype(int), 0, h - 1)
        c2 = np.clip(((uu - C.X0) / (F * C.PAS)).astype(int), 0, w - 1)
        hors_carte = uu < C.X0                                                # voisin à gauche de la carte = texte
        risque |= texte[r2, c2] | hors_carte
    risque = ndi.binary_dilation(risque, iterations=int(1.0 / MM))
    return valide, texte, risque


def fond(a, masque):
    """Fond lent : médiane sur ~12 mm, calculée à basse résolution."""
    g = 8
    h, w = a.shape[0] // g, a.shape[1] // g
    b = np.where(masque, a, np.nan)[:h * g, :w * g].reshape(h, g, w, g)
    with np.errstate(all="ignore"):
        petit = np.nanmedian(b, axis=(1, 3))
    trou = ~np.isfinite(petit)
    idx = ndi.distance_transform_edt(trou, return_distances=False, return_indices=True)
    petit = ndi.median_filter(petit[tuple(idx)], size=39)
    return ndi.zoom(petit, (a.shape[0] / h, a.shape[1] / w), order=1)[:a.shape[0], :a.shape[1]]


def score_boites(r, propre, hauteur, largeur):
    """z de la moyenne écrêtée dans chaque rectangle hauteur x largeur (mm), centré sur le pixel."""
    taille = (max(1, int(hauteur / MM)), max(1, int(largeur / MM)))
    num = ndi.uniform_filter(np.where(propre, r, 0).astype(np.float32), taille, mode="constant")
    den = ndi.uniform_filter(propre.astype(np.float32), taille, mode="constant")
    moy = np.where(den > 0.7, num / np.maximum(den, 1e-6), np.nan)
    v = moy[np.isfinite(moy)]
    med = np.median(v); sig = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (moy - med) / sig, med, sig


def main():
    ls, la = logit_reduit("student_propre.npy"), logit_reduit("avril_propre.npy")
    valide, texte, risque = masques(ls.shape)
    propre = valide & ~texte & ~risque
    print("validité %.0f %% | texte %.0f %% | risque de fantôme %.0f %% de la carte" % (100 * valide.mean(), 100 * texte.mean(), 100 * risque.mean()))
    print("carte réduite", ls.shape, "| zone examinée : %.0f cm² (%.0f %% de la zone vierge valide ; le reste fait face à du texte)"
          % (propre.sum() * MM * MM / 100, 100 * propre.sum() / max((valide & ~texte).sum(), 1)))
    r = {}
    for nom, a in (("student", ls), ("avril", la)):
        # « évidence d'encre » : 0 sous le logit BAS (papyrus nu), plafonnée au logit HAUT (encre franche)
        r[nom] = (np.clip(a, BAS, HAUT) - BAS).astype(np.float32)
    r["moyenne"] = (r["student"] + r["avril"]) / 2
    np.savez_compressed(D / "logits_reduits.npz", student=ls, avril=la, propre=propre, valide=valide, texte=texte, residu=r["moyenne"])
    meilleur = np.full(ls.shape, -np.inf, np.float32); quelle = np.zeros(ls.shape, np.int8)
    tailles = [(h, w) for h in HAUTEURS for w in LARGEURS]
    reference = {}
    for k, (hh, ww) in enumerate(tailles):
        z, med, sig = score_boites(r["moyenne"], propre, hh, ww)
        reference[f"{hh}x{ww}"] = (float(med), float(sig))
        z = np.where(np.isfinite(z), z, -np.inf)
        m = z > meilleur; meilleur[m] = z[m]; quelle[m] = k
    np.save(D / "score_faible.npy", meilleur)
    fini = meilleur[np.isfinite(meilleur)]
    print("distribution du meilleur z dans la zone examinée : médiane %.1f | 99 %% %.1f | 99,9 %% %.1f | max %.1f"
          % (np.median(fini), np.percentile(fini, 99), np.percentile(fini, 99.9), fini.max()))
    # référence : le même score sur une vraie ligne de texte (dernière colonne, non masquée)
    tout = valide
    zt, _, _ = score_boites(r["moyenne"], tout, 3.0, 12.0)
    med, sig = reference["3.0x12.0"]
    num = ndi.uniform_filter(np.where(tout, r["moyenne"], 0), (int(3.0 / MM), int(12.0 / MM)), mode="constant")
    zone_texte = np.zeros(ls.shape, bool); zone_texte[int(5 / MM):int(40 / MM), col_de(222):col_de(275)] = True
    print("pour comparaison, vraies lignes de la dernière colonne (rectangle 3 x 12 mm) : z médian %.0f, max %.0f"
          % (np.median((num[zone_texte] - med) / sig), ((num[zone_texte] - med) / sig).max()))
    # candidats : maxima successifs, en écartant le voisinage de chaque maximum
    s = meilleur.copy(); cands = []
    for _ in range(40):
        i = int(np.argmax(s)); y, x = divmod(i, s.shape[1])
        if not np.isfinite(s[y, x]):
            break
        hh, ww = tailles[quelle[y, x]]
        a, b = int(hh / MM / 2), int(ww / MM / 2)
        boite = (slice(max(0, y - a), y + a + 1), slice(max(0, x - b), x + b + 1))
        zs = float(np.nanmean(np.where(propre[boite], r["student"][boite], np.nan)))
        za = float(np.nanmean(np.where(propre[boite], r["avril"][boite], np.nan)))
        cands.append(dict(z=float(s[y, x]), x_mm=float(xmm(x)), y_mm=float(y * MM), hauteur_mm=hh, largeur_mm=ww,
                          student=zs, avril=za, y=int(y), x=int(x)))
        s[max(0, y - 2 * a - 40):y + 2 * a + 41, max(0, x - b - 60):x + b + 61] = -np.inf
    (D / "candidats_faibles.json").write_text(json.dumps(cands, indent=1))
    print("  #     z   x mm   y mm  rectangle (mm)  évidence moyenne student / avril")
    for i, c in enumerate(cands[:20], 1):
        print("%3d %5.1f %6.1f %6.1f   %3.1f x %4.1f      %+.2f / %+.2f" % (i, c["z"], c["x_mm"], c["y_mm"], c["hauteur_mm"], c["largeur_mm"], c["student"], c["avril"]))


if __name__ == "__main__":
    main()
