#!/usr/bin/env python3
"""Visionneuse de toute la zone vierge de la fin de PHerc. Paris 4 (x 281-408 mm, toute la hauteur),
plus une planche d'exemple : à quoi ressemblerait un titre dans la carte du modèle ink_9um.

  .venv/bin/python -I scripts/visionneuse_zone.py
Sorties : apercus/zone_vierge/index.html (+ calques), apercus/exemple_titre_dans_vierge.png
"""
import glob
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import comparaison_titre as K  # noqa: E402
import volume as V  # noqa: E402

PX = 0.0096
ZONE = (0.0, 167.4, 281.0, 408.0)
DOS = V.RACINE / "apercus/zone_vierge"
I9 = V.DATA / "ink9"


def canonique_reduit(z, f=2):
    """Carte « canonical » ramenée directement à 9,6 µm x f, bande par bande (le fichier fait 12 Go décompressé)."""
    r0, r1, c0, c1 = K.etendue(z); k = 4 * f
    tif = glob.glob(str(K.SEG / "20260701183124-w010-027/ink-detection/*new_canon*.tif"))[0]
    h, w = (r1 - r0) // f, (c1 - c0) // f
    out = np.zeros((h, w), np.float32)
    for r in range(0, h, 256):
        a = V.lire_tif_region(tif, ((r0 // 1 + r * f) * 4, (r0 + (r + 256) * f) * 4), (c0 * 4, c0 * 4 + w * k))
        n = min(a.shape[0] // k, h - r)
        out[r:r + n] = a[:n * k].reshape(n, k, w, k).mean((1, 3))
    return np.clip((out - 32) / (230 - 32), 0, 1)


def calques():
    r0, r1, c0, c1 = K.etendue(ZONE)
    a9 = np.load(I9 / "p4_vierge_d+0.npy", mmap_mode="r"); H, W = a9.shape[1:]
    z = (ZONE[0], H * PX, ZONE[2], ZONE[2] + W * PX)
    yield "1. Surface du papyrus (scan brut)", K.reduire(K.surface("p4_vierge", 0, 281, z), 2)
    yield "2. Modèle 3D d'avril", K.reduire(K.carte3d("avril_propre.npy", z), 2)
    yield "3. Modèle 3D de septembre", K.reduire(K.carte3d("student_propre.npy", z), 2)
    yield "4. Modèle 2D « canonical »", canonique_reduit(z)
    yield "5. ink_9um (multi-rouleaux)", K.reduire(K.ink9("p4_vierge", 0, 281, z), 2)


HTML = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Zone vierge de PHerc. Paris 4</title>
<style>
  html,body{height:100%}
  body{margin:0;background:#141414;color:#e8e8e8;font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif;overflow:hidden;display:flex;flex-direction:column}
  #barre{padding:8px 12px;background:#1f1f1f;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:6px 10px;font-size:14px;cursor:pointer}
  button.actif{background:#c79a2b;color:#111;border-color:#c79a2b}
  #info{margin-left:auto;font-variant-numeric:tabular-nums;color:#ffd978}
  #aide{padding:4px 12px 8px;color:#aaa;font-size:13px;background:#1f1f1f}
  #vue{position:relative;flex:1;overflow:hidden;cursor:grab}
  #vue img{position:absolute;top:0;left:0;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
</style></head><body>
<div id="barre"></div>
<div id="aide">Touches 1 à N : changer de technique au même endroit · molette : zoom · glisser : déplacer · position du curseur en haut à droite (à me donner si tu vois quelque chose)</div>
<div id="vue"><img id="img" alt=""></div>
<script>
const noms=NOMS,X0=XZERO,Y0=YZERO,MM=0.0192;
const barre=document.getElementById('barre'),img=document.getElementById('img'),vue=document.getElementById('vue');
let z=0.14,tx=20,ty=10;
noms.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>choisir(i);barre.appendChild(b)});
const info=document.createElement('span');info.id='info';barre.appendChild(info);
function choisir(i){img.src='calque'+(i+1)+'.jpg';[...barre.querySelectorAll('button')].forEach((b,j)=>b.classList.toggle('actif',j===i))}
function poser(){img.style.transform=`translate(${tx}px,${ty}px) scale(${z})`}
addEventListener('keydown',e=>{const n=parseInt(e.key);if(n>=1&&n<=noms.length)choisir(n-1)});
vue.addEventListener('wheel',e=>{e.preventDefault();const r=vue.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.15:1/1.15;tx=mx-(mx-tx)*f;ty=my-(my-ty)*f;z*=f;poser()},{passive:false});
let d=null;vue.addEventListener('mousedown',e=>{d=[e.clientX-tx,e.clientY-ty];vue.style.cursor='grabbing'});
addEventListener('mouseup',()=>{d=null;vue.style.cursor='grab'});
addEventListener('mousemove',e=>{if(d){tx=e.clientX-d[0];ty=e.clientY-d[1];poser()}
  const r=vue.getBoundingClientRect(),px=(e.clientX-r.left-tx)/z,py=(e.clientY-r.top-ty)/z;
  info.textContent='x '+(X0+px*MM).toFixed(1)+' mm   y '+(Y0+py*MM).toFixed(1)+' mm'});
choisir(noms.length-1);poser();
</script></body></html>"""


def exemple():
    """Le titre de PHerc. 139 (vu par ink_9um) collé dans un morceau de papyrus vierge de Paris 4, à trois intensités."""
    t = (np.load(I9 / "titre139_seed42_075000.npy").astype(np.float32) + np.load(I9 / "titre139_seed43_075000.npy")) / 2 / 255
    t = t[150:1450, 800:4400]                                              # 12,5 x 34,6 mm autour du titre
    a = np.load(I9 / "p4_vierge_d+0.npy", mmap_mode="r")
    r0, c0 = int(70 / PX), int((300 - 281) / PX)                            # vierge de Paris 4 : y 70 mm, x 300 mm
    fond = np.asarray(a[:, r0:r0 + t.shape[0], c0:c0 + t.shape[1]]).astype(np.float32).mean(0) / 255
    f = 2; h, w = t.shape[0] // f, t.shape[1] // f
    try:
        po = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24); po2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except Exception:
        po = po2 = ImageFont.load_default()
    lignes = [("Papyrus vierge de Paris 4, tel que le modèle le voit (aucun titre)", fond)]
    for k in (1.0, 0.5, 0.25):
        lignes.append((f"EXEMPLE FABRIQUÉ : le titre de PHerc. 139 collé dans ce même vierge, à {int(100 * k)} % de sa force", np.maximum(fond, 0.25 + k * (t - 0.25))))
    im = Image.new("RGB", (w + 40, 70 + len(lignes) * (h + 46)), (18, 18, 18)); d = ImageDraw.Draw(im)
    d.text((20, 16), "Ce qu'il faut chercher : des lignes de lettres d'environ 4 mm de haut, comme ici", fill=(255, 220, 120), font=po2)
    for i, (titre, m) in enumerate(lignes):
        py = 70 + i * (h + 46); d.text((20, py + 6), titre, fill=(140, 220, 255) if i == 0 else (255, 150, 150), font=po)
        v = (255 * np.clip((m - 0.25) / 0.65, 0, 1)).astype(np.uint8)
        im.paste(Image.fromarray(v).resize((w, h), Image.LANCZOS), (20, py + 38))
    im.save(V.RACINE / "apercus/exemple_titre_dans_vierge.png"); print("exemple", im.size)


def main():
    DOS.mkdir(parents=True, exist_ok=True); noms = []
    for i, (nom, m) in enumerate(calques()):
        Image.fromarray((255 * np.clip(m, 0, 1)).astype(np.uint8)).save(DOS / f"calque{i + 1}.jpg", quality=86); noms.append(nom); print(nom, m.shape, flush=True)
    import json
    (DOS / "index.html").write_text(HTML.replace("NOMS", json.dumps(noms, ensure_ascii=False)).replace("XZERO", str(ZONE[2])).replace("YZERO", str(ZONE[0])))
    exemple()


if __name__ == "__main__":
    main()
