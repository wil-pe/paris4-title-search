#!/usr/bin/env python3
"""Détecteur d'écriture pâle sur les cartes d'encre publiées (scan 2,4 µm) de toute la fin de PHerc. Paris 4.

Aucun téléchargement : tout part des cartes déjà sur disque (data/PHercParis4/carte), qui couvrent
x 172-408 mm sur toute la hauteur du segment, donc aussi les endroits que le scan 1,129 µm ne voit pas.

Principe : un petit réseau apprend à distinguer du papyrus vierge du même papyrus auquel on superpose
de vraies lettres du scribe, affaiblies de « delta » logits (delta 2,5 / 3,7 / 4,5 correspond, d'après
l'étalonnage, à une encre à 60 / 40 / 0 % de contraste). Deux précautions :
  - les lettres qui servent au test (colonne précédente) ne sont jamais vues à l'apprentissage
    (dernière colonne) ;
  - la zone vierge est coupée en bandes de 8 mm ; chaque bande est notée par un réseau qui ne l'a
    jamais vue comme exemple « vierge » (sinon un titre pâle présent dans les exemples négatifs
    serait appris comme du vide).

  .venv/bin/python -I scripts/detecteur_carte.py            # cartes des deux modèles 3D (scan 2,4 µm)
  .venv/bin/python -I scripts/detecteur_carte.py canon      # carte 2D du modèle « canonical » (celui qui lit le titre de PHerc. 139)
  .venv/bin/python -I scripts/detecteur_carte.py ink9       # cartes du modèle multi-rouleaux ink_9um (scripts/ink9.py)
  .venv/bin/python -I scripts/detecteur_carte.py fin2d      # carte 2D d'un autre modèle, calculée sur le scan 1,129 µm (segment jumeau)
"""
import glob
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import volume as V  # noqa: E402

D = C.SORTIE
G = 2                                  # réduction : 19,2 µm par pixel
MM = 0.0024 * C.PAS * G
W = 256                                # fenêtre : 4,9 mm
PAS_F = 64                             # pas du balayage : 1,2 mm
BANDE = 8.0                            # mm : largeur des bandes de validation croisée
DELTAS = (0.0, 2.5, 3.7, 4.2, 4.5, 5.5)
APERCUS = pathlib.Path(__file__).resolve().parent.parent / "apercus"
rng = np.random.default_rng(0)


def logit_reduit(nom):
    """Carte de scores 0..255 -> logits moyennés par blocs de G x G pixels."""
    a = np.load(D / nom, mmap_mode="r")
    v = np.arange(256, dtype=np.float32)
    lut = np.log((v + 0.5) / (255.5 - v)).astype(np.float32)
    h, w = a.shape[0] // G, a.shape[1] // G
    out = np.empty((h, w), np.float32)
    for r in range(0, h, 500):
        bloc = lut[np.asarray(a[r * G:(r + 500) * G, :w * G])]
        out[r:r + 500] = bloc.reshape(-1, G, w, G).mean((1, 3))
    return out


LUT = np.log((np.arange(256, dtype=np.float32) + 0.5) / (255.5 - np.arange(256, dtype=np.float32))).astype(np.float32)
SEG = V.DATA / "PHercParis4/segments"


def charger_canon():
    """Carte 2D « canonical » du segment 701 (2,4 µm/px), ramenée à 19,2 µm ; (cartes, couvert)."""
    tif = glob.glob(str(SEG / "20260701183124-w010-027/ink-detection/*new_canon*.tif"))[0]
    h, w = C.Y1 // (C.PAS * G), (C.X1 - C.X0) // (C.PAS * G); k = C.PAS * G
    out = np.full((h, w), LUT[0], np.float32); couv = np.zeros((h, w), np.float32)
    for r in range(0, h, 256):
        a = V.lire_tif_region(tif, (r * k, min((r + 256) * k, C.Y1)), (C.X0, C.X0 + w * k))
        n = a.shape[0] // k
        out[r:r + n] = LUT[a[:n * k]].reshape(n, k, w, k).mean((1, 3)); couv[r:r + n] = (a[:n * k] > 0).reshape(n, k, w, k).mean((1, 3))
    return out[None], couv > 0.99


def charger_fin2d():
    """Carte 2D calculée sur le scan 1,129 µm (segment jumeau 623, 2,258 µm/px), replacée dans le repère du segment 701."""
    tif = glob.glob(str(SEG / "20260623141924-w010-027/ink-detection/*1.129um*.tif"))[0]
    import tifffile
    with tifffile.TiffFile(tif) as t:
        H, Wd = t.pages[0].shape
    ech = 170160 / Wd                                        # 2,258 µm -> 2,4 µm
    h, w = C.Y1 // (C.PAS * G), (C.X1 - C.X0) // (C.PAS * G)
    x_dep = int(round((C.X0 - 40) / ech))                    # colonne du TIFF correspondant à x = X0 du segment 701
    lignes = []; couvs = []
    for r in range(0, H, 2048):
        a = V.lire_tif_region(tif, (r, min(r + 2048, H)), (x_dep, Wd))
        n, m = a.shape[0] // 8, a.shape[1] // 8
        lignes.append(LUT[a[:n * 8, :m * 8]].reshape(n, 8, m, 8).mean((1, 3))); couvs.append((a[:n * 8, :m * 8] > 0).reshape(n, 8, m, 8).mean((1, 3)))
    b = ndi.zoom(np.concatenate(lignes), ech, order=1); cv = ndi.zoom(np.concatenate(couvs).astype(np.float32), ech, order=1)
    out = np.full((h, w), LUT[0], np.float32); couv = np.zeros((h, w), np.float32)
    r0 = 4416 // (C.PAS * G); n, m = min(b.shape[0], h - r0), min(b.shape[1], w)
    out[r0:r0 + n, :m] = b[:n, :m]; couv[r0:r0 + n, :m] = cv[:n, :m]
    return out[None], couv > 0.99


def charger_ink9():
    """Cartes du modèle multi-rouleaux ink_9um calculées par scripts/ink9.py (9,6 µm, moyenne des deux jeux de poids).
    La sortie du modèle va de 0,25 (pas d'encre) à 0,75 (encre) : on la ramène à 0..1 avant de passer en logit."""
    h, w = C.Y1 // (C.PAS * G), (C.X1 - C.X0) // (C.PAS * G)
    out = np.full((h, w), LUT[0], np.float32); couv = np.zeros((h, w), bool)
    v = np.arange(256, dtype=np.float32) / 255
    q = np.clip((v - 0.25) / 0.5, 0.002, 0.998); lut9 = np.log(q / (1 - q)).astype(np.float32)
    for nom, y0, x0 in (("p4_vierge", 0.0, 281.0), ("p4_sous_colonne", 44.0, 205.0), ("p4_texte", 5.0, 215.0)):
        chemin = V.DATA / "ink9" / f"{nom}_d+0.npy"
        if not chemin.exists():
            continue
        a = np.load(chemin, mmap_mode="r"); n, m = a.shape[1] // G, a.shape[2] // G
        r0 = int(round(y0 / MM)); c0 = col_de(x0)
        for r in range(0, n, 512):
            bloc = np.asarray(a[:, r * G:(r + 512) * G, :m * G]); k = bloc.shape[1] // G
            lg = lut9[bloc[:, :k * G]].mean(0).reshape(k, G, m, G).mean((1, 3)); fait = (bloc[:, :k * G] > 0).all(0).reshape(k, G, m, G).all((1, 3))
            rr, cc = min(k, h - r0 - r), min(m, w - c0)
            if rr <= 0 or cc <= 0:
                continue
            out[r0 + r:r0 + r + rr, c0:c0 + cc] = np.where(fait[:rr, :cc], lg[:rr, :cc], out[r0 + r:r0 + r + rr, c0:c0 + cc])
            couv[r0 + r:r0 + r + rr, c0:c0 + cc] |= fait[:rr, :cc]
    return out[None], couv


# zones de vraies lettres (mm) : (y0, y1, x0, x1) pour apprendre puis pour tester ; affaiblissements essayés
SOURCES = {
    "3d": dict(app=(0, 44, 212, 285), test=(10, 150, 173, 198), deltas=(0.0, 2.5, 3.7, 4.5, 5.5, 6.5, 7.5), dmax=6.5),
    "canon": dict(app=(0, 44, 212, 285), test=(10, 150, 173, 198), deltas=(0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.5), dmax=4.0),
    "ink9": dict(app=(5, 25, 215, 250), test=(25, 45, 215, 250), deltas=(0.0, 2.0, 3.5, 5.0, 6.0, 7.0, 8.5), dmax=7.5),
    "fin2d": dict(app=(44, 80, 173, 198), test=(80, 112, 173, 198), deltas=(0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.5), dmax=4.0),
}


def xmm(col):
    return (C.X0 + col * G * C.PAS) * 0.0024


def col_de(x_mm):
    return int((x_mm / 0.0024 - C.X0) / (G * C.PAS))


def norm(a):
    return (np.clip(a, -6, 4) / 3).astype(np.float32)


def reseau(n=2):
    def bloc(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(4, o), nn.ReLU())
    return nn.Sequential(bloc(n, 16), bloc(16, 32), bloc(32, 64), bloc(64, 64), bloc(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))


def positions(masque, seuil):
    """Coins (r, c) des fenêtres de la grille dont la part de `masque` atteint `seuil`."""
    part = ndi.uniform_filter(masque.astype(np.float32), W, mode="constant")[W // 2::PAS_F, W // 2::PAS_F]
    rr, cc = np.nonzero(part >= seuil)
    ok = (rr * PAS_F + W <= masque.shape[0]) & (cc * PAS_F + W <= masque.shape[1])
    return np.stack([rr[ok] * PAS_F, cc[ok] * PAS_F], 1)


def parite(c):
    """Bande de validation croisée (0 ou 1) d'une fenêtre dont le coin gauche est à la colonne c."""
    return (np.floor(xmm(np.asarray(c) + W / 2) / BANDE) % 2).astype(int)


class Texte:
    """Réservoir de fenêtres de vraies lettres, prises dans une zone de la carte."""

    def __init__(self, cartes, r0, r1, c0, c1, seuil=0.0):
        self.z = cartes[:, r0:r1, c0:c1]; self.seuil = seuil
        self.encre = (self.z > self.seuil).any(0)
        dens = ndi.uniform_filter(self.encre.astype(np.float32), W // 2, mode="constant")
        rr, cc = np.nonzero(dens[::16, ::16] > 0.05)
        self.centres = np.stack([rr * 16, cc * 16], 1)

    def tirer(self, echelle):
        h = int(W / echelle)
        for _ in range(100):
            r, c = self.centres[rng.integers(len(self.centres))] - h // 2
            if 0 <= r <= self.z.shape[1] - h and 0 <= c <= self.z.shape[2] - h:
                break
        else:
            r, c = 0, 0
        t = self.z[:, r:r + h, c:c + h]; m = ndi.binary_dilation(self.encre[r:r + h, c:c + h], iterations=2)
        t = np.where(m[None], t, -30)
        if h != W:
            t = np.stack([np.pad(ndi.zoom(x, W / h, order=1)[:W, :W], 0) for x in t])
            t = np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, W - t.shape[2])), constant_values=-30)
        return t


def lot(cartes, coins, texte, n, deltas=None, dmax=6.0):
    """n fenêtres : indices pairs = vierges, impairs = vierges + lettres affaiblies."""
    x = np.empty((n, cartes.shape[0], W, W), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        r, c = coins[rng.integers(len(coins))]
        f = cartes[:, r:r + W, c:c + W]
        if rng.random() < 0.5:
            f = f[:, :, ::-1]
        if i % 2:
            d = rng.uniform(0, dmax) if deltas is None else deltas[rng.integers(len(deltas))]
            t = texte.tirer(float(rng.choice([1.0, 1.0, 1.3, 1.6])))
            f = np.maximum(f, np.where(t > -20, t - d, -30)); y[i] = 1
        x[i] = norm(f)
    return torch.from_numpy(x), torch.from_numpy(y)


def noter(net, cartes, coins, taille=256):
    out = np.empty(len(coins), np.float32)
    with torch.inference_mode():
        for i in range(0, len(coins), taille):
            x = np.stack([norm(cartes[:, r:r + W, c:c + W]) for r, c in coins[i:i + taille]])
            out[i:i + taille] = net(torch.from_numpy(x)).squeeze(1).numpy()
    return out


def planche(cartes, cands, nom, n=16):
    """Vignettes de 6 mm autour des meilleurs endroits (cartes réduites à 19,2 µm, un canal par colonne)."""
    cote, cols = 312, 4; js = cands[:n]; lignes = -(-len(js) // cols); nc = cartes.shape[0]
    im = Image.new("RGB", (cols * (nc * (cote + 2) + 8), lignes * (cote + 24)), (20, 20, 20)); d = ImageDraw.Draw(im)
    lo, hi = np.percentile(cartes[:, ::20, ::20], 50), np.percentile(cartes[:, ::20, ::20], 99.9)
    for i, k in enumerate(js):
        r = int(k["y_mm"] / MM); c = col_de(k["x_mm"]); px, py = (i % cols) * (nc * (cote + 2) + 8), (i // cols) * (cote + 24)
        for j in range(nc):
            v = np.full((cote, cote), lo, np.float32); m = cartes[j]
            r0, r1, c0, c1 = max(r - cote // 2, 0), min(r + cote // 2, m.shape[0]), max(c - cote // 2, 0), min(c + cote // 2, m.shape[1])
            v[r0 - (r - cote // 2):r1 - (r - cote // 2), c0 - (c - cote // 2):c1 - (c - cote // 2)] = m[r0:r1, c0:c1]
            im.paste(Image.fromarray((255 * np.clip((v - lo) / (hi - lo), 0, 1)).astype(np.uint8)), (px + j * (cote + 2), py + 22))
        d.text((px + 4, py + 5), f"#{i + 1} score {k['score']:+.1f}  x {k['x_mm']:.1f}  y {k['y_mm']:.1f} mm  fantôme {100 * k['risque']:.0f} %", fill=(255, 220, 120))
    im.save(APERCUS / f"paris4_candidats_carte{nom}.png")


def main(source="3d"):
    torch.set_num_threads(6)
    P = SOURCES[source]; suf = "" if source == "3d" else "_" + source
    if source == "3d":
        cartes = np.stack([logit_reduit("student_propre.npy"), logit_reduit("avril_propre.npy")]); couvert = np.ones(cartes.shape[1:], bool)
    else:
        cartes, couvert = {"canon": charger_canon, "fin2d": charger_fin2d, "ink9": charger_ink9}[source]()
    h, w = cartes.shape[1:]
    z = np.load(D / "logits_reduits.npz")                                   # masques à 38,4 µm, calculés par recherche_faible.py
    haut = lambda m: np.repeat(np.repeat(m, 2, 0), 2, 1)[:h, :w]            # noqa: E731
    valide, texte, propre = haut(z["valide"]) & couvert, haut(z["texte"]), haut(z["propre"]) & couvert
    risque = valide & ~texte & ~propre
    fond = float(np.median(cartes[0][propre][::20]))
    print(f"source {source} : carte {cartes.shape} à {1000 * MM:.1f} µm ; zone vierge couverte : {(valide & ~texte).sum() * MM * MM / 100:.0f} cm², dont {risque.sum() * MM * MM / 100:.0f} cm² face à du texte ; logit médian du vierge {fond:.2f}", flush=True)
    seuil_encre = 0.0 if source == "3d" else fond + 2.0                      # les cartes 2D ont un plancher vers -2
    zone = lambda q: (int(q[0] / MM), int(q[1] / MM), max(col_de(q[2]), 0), col_de(q[3]))   # noqa: E731
    t_app = Texte(cartes, *zone(P["app"]), seuil=seuil_encre); t_test = Texte(cartes, *zone(P["test"]), seuil=seuil_encre)
    print(f"lettres : {len(t_app.centres)} positions pour apprendre, {len(t_test.centres)} pour tester ; logit médian de l'encre {np.median(t_test.z[0][t_test.encre]):.2f}", flush=True)
    if source != "3d":
        cartes = np.where(couvert[None], cartes, fond).astype(np.float32)      # pas de bord franc entre zone calculée et trou
    vierges = positions(propre, 0.97)
    grille = positions(valide & ~texte, 0.7 if source == "3d" else 0.97)     # cartes 2D : fenêtres entièrement couvertes seulement
    print(f"fenêtres vierges sûres : {len(vierges)} ; fenêtres à noter : {len(grille)}", flush=True)
    scores = np.full(len(grille), np.nan, np.float32); bilan = {}; seuils = []
    for pli in (0, 1):
        app = vierges[parite(vierges[:, 1]) == pli]; test = vierges[parite(vierges[:, 1]) != pli]
        net = reseau(cartes.shape[0]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3); perte = nn.BCEWithLogitsLoss()
        net.train()
        for pas in range(1500):
            for g in opt.param_groups:
                g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * pas / 1500))
            x, y = lot(cartes, app, t_app, 48, dmax=P["dmax"])
            opt.zero_grad(); p = perte(net(x).squeeze(1), y); p.backward(); opt.step()
        net.eval()
        neg = noter(net, cartes, test)
        seuil = float(np.quantile(neg, 0.999)); seuils.append(seuil)
        print(f"réseau {pli} : {len(app)} fenêtres vierges vues, {len(test)} jamais vues ; vierge : médiane {np.median(neg):.2f}, 99,9 % {seuil:.2f}, max {neg.max():.2f}")
        with torch.inference_mode():
            for d in P["deltas"]:
                pos = np.concatenate([net(lot(cartes, test, t_test, 256, (d,))[0][1::2]).squeeze(1).numpy() for _ in range(3)])
                bilan.setdefault(f"{d}", []).append(float((pos > seuil).mean()))
                print(f"   lettres affaiblies de {d:.1f} logits : {100 * (pos > seuil).mean():.0f} % des fenêtres détectées", flush=True)
        m = parite(grille[:, 1]) != pli
        scores[m] = noter(net, cartes, grille[m]) - seuil                    # score ramené au seuil du réseau qui a noté
        torch.save(net.state_dict(), D / f"detecteur_carte{suf}_{pli}.pt")
    part_risque = np.array([risque[r:r + W, c:c + W].mean() for r, c in grille])
    sur = part_risque < 0.05
    print(f"fenêtres notées : {len(grille)} ; au-dessus du seuil : {(scores > 0).sum()}, dont {((scores > 0) & sur).sum()} hors des zones à fantômes "
          f"(attendu par hasard : {sur.sum() / 1000:.0f})")
    ordre = np.argsort(-scores); cands = []
    for i in ordre:
        r, c = grille[i]; k = dict(score=float(scores[i]), x_mm=float(xmm(c + W / 2)), y_mm=float((r + W / 2) * MM), risque=float(part_risque[i]))
        if all(abs(k["x_mm"] - g["x_mm"]) > 4 or abs(k["y_mm"] - g["y_mm"]) > 4 for g in cands):
            cands.append(k)
        if len(cands) == 40:
            break
    np.savez_compressed(D / f"score_detecteur{suf}.npz", grille=grille, scores=scores, risque=part_risque)
    (D / f"candidats_detecteur{suf}.json").write_text(json.dumps(dict(bilan=bilan, seuils=seuils, candidats=cands), indent=1))
    print("   score au-dessus du seuil   x mm   y mm   part face à du texte")
    for k in cands[:20]:
        print("   %+6.2f                 %6.1f %6.1f   %3.0f %%" % (k["score"], k["x_mm"], k["y_mm"], 100 * k["risque"]))
    planche(cartes, cands, suf)
    surs = [k for k in cands if k["risque"] < 0.05]
    planche(cartes, surs, suf + "_hors_fantomes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3d")
