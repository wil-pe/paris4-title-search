#!/usr/bin/env python3
"""Visionneuse « page par page » de la zone vierge de la fin de PHerc. Paris 4, pour un contrôle à l'œil.

Mise en page : à gauche la liste des pages (comme dans Aperçu sur Mac), au centre
la page à examiner en grand, à droite deux ou trois exemples fixes de ce qu'il faut chercher. Tout ce
qui sert à situer une capture d'écran est affiché dans l'image : technique, page, règles en millimètres.

  .venv/bin/python -I scripts/visionneuse_pages.py
Sortie : apercus/visionneuse/index.html (+ pages/, exemples/, donnees.js)
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import comparaison_titre as K  # noqa: E402
import visionneuse_zone as Z  # noqa: E402
import volume as V  # noqa: E402

MM = 0.0192                                  # mm par pixel des pages
LARG, HAUT = 40.0, 30.0                      # taille d'une page en mm
DOS = V.RACINE / "apercus/visionneuse"
TEXTE = (24.5, 39.5, 219.0, 245.0)           # vrai texte de Paris 4 pour les exemples : deux lignes de la dernière colonne
VIERGE = (70.0, 85.0, 300.0, 326.0)          # papyrus vierge sur lequel on colle l'exemple


def debuts(a, b, taille, pas):
    v = list(np.arange(a, b - taille, pas)) + [b - taille]
    return sorted({round(float(x), 1) for x in v})


def source(i):
    """Fonction (zone en mm) -> image 0..1 à 9,6 µm, pour la technique i, sur n'importe quelle zone couverte."""
    if i == 0:
        return lambda z: K.surface("p4_texte", 5, 215, z) if z[2] < 281 else K.surface("p4_vierge", 0, 281, z)
    if i == 1:
        return lambda z: K.carte3d("avril_propre.npy", z)
    if i == 2:
        return lambda z: K.carte3d("student_propre.npy", z)
    if i == 3:
        return K.canonique
    return lambda z: K.ink9("p4_texte", 5, 215, z) if z[2] < 281 else K.ink9("p4_vierge", 0, 281, z)


def enregistrer(a, chemin, q=85):
    Image.fromarray((255 * np.clip(a, 0, 1)).astype(np.uint8)).save(chemin, quality=q)


ZONES = [   # nom affiché, (y0, y1, x0, x1) en mm, préfixe des fichiers de scripts/ink9.py, origine (y, x) de ces fichiers
    ("après la dernière colonne", (0.0, 167.4, 281.0, 408.0), "p4_vierge", (0.0, 281.0)),
    ("sous la dernière colonne", (44.0, 167.4, 205.0, 281.0), "p4_sous_colonne", (44.0, 205.0)),
]
NOMS = ["1. Surface du papyrus (scan brut)", "2. Modèle 3D d'avril", "3. Modèle 3D de septembre", "4. Modèle 2D « canonical »", "5. ink_9um (multi-rouleaux)"]


def carte_zone(i, zone, prefixe, origine):
    """Carte entière de la zone, à 19,2 µm, pour la technique i."""
    a = np.load(V.DATA / "ink9" / f"{prefixe}_d+0.npy", mmap_mode="r")
    z = (zone[0], zone[0] + a.shape[1] * 0.0096, zone[2], zone[2] + a.shape[2] * 0.0096)
    if i == 0:
        return K.reduire(K.surface(prefixe, origine[0], origine[1], z), 2)
    if i == 1:
        return K.reduire(K.carte3d("avril_propre.npy", z), 2)
    if i == 2:
        return K.reduire(K.carte3d("student_propre.npy", z), 2)
    if i == 3:
        return Z.canonique_reduit(z)
    return K.reduire(K.ink9(prefixe, origine[0], origine[1], z), 2)


def main():
    (DOS / "pages").mkdir(parents=True, exist_ok=True); (DOS / "exemples").mkdir(exist_ok=True)
    pages = None
    for i, nom in enumerate(NOMS):
        liste = []
        for titre, zone, prefixe, origine in ZONES:
            if not (V.DATA / "ink9" / f"{prefixe}_d+0.npy").exists():
                continue
            carte = carte_zone(i, zone, prefixe, origine)
            h_mm, w_mm = carte.shape[0] * MM, carte.shape[1] * MM
            for y in debuts(0, h_mm, HAUT, HAUT - 2):
                for x in debuts(0, w_mm, LARG, LARG - 2):
                    k = len(liste); r, c = int(round(y / MM)), int(round(x / MM))
                    enregistrer(carte[r:r + int(HAUT / MM), c:c + int(LARG / MM)], DOS / "pages" / f"t{i + 1}_p{k + 1:02d}.jpg")
                    liste.append(dict(y=zone[0] + y, x=zone[2] + x, zone=titre))
        pages = liste
        # exemples fixes : du vrai texte de Paris 4 vu par cette technique, collé dans du vierge à trois forces
        lire = source(i)
        t = K.reduire(lire(TEXTE), 2); f = K.reduire(lire(VIERGE), 2)
        h, w = min(t.shape[0], f.shape[0]), min(t.shape[1], f.shape[1]); t, f = t[:h, :w], f[:h, :w]
        if i == 0:
            enregistrer(t, DOS / "exemples" / f"t{i + 1}_e1.jpg"); enregistrer(f, DOS / "exemples" / f"t{i + 1}_e2.jpg")
        else:
            for j, force in enumerate((1.0, 0.5, 0.25)):
                enregistrer(np.maximum(f, force * t), DOS / "exemples" / f"t{i + 1}_e{j + 1}.jpg")
        print(nom, "->", len(pages), "pages", flush=True)
    donnees = dict(noms=NOMS, mm=MM, larg=LARG, haut=HAUT, pages=pages,
                   exemples=[["vrai texte du rouleau : invisible sur le scan brut", "papyrus vierge, pour comparer"]]
                   + [["vrai texte collé dans du vierge — 100 %", "le même à 50 %", "le même à 25 %"]] * (len(NOMS) - 1))
    (DOS / "donnees.js").write_text("const D = " + json.dumps(donnees, ensure_ascii=False) + ";\n")
    (DOS / "index.html").write_text(HTML)
    print("visionneuse écrite :", DOS / "index.html")


HTML = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>PHerc. Paris 4 — zone du titre, contrôle à l'œil</title>
<style>
  html,body{height:100%}
  body{margin:0;background:#141414;color:#e8e8e8;font:14px/1.35 -apple-system,Helvetica,Arial,sans-serif;overflow:hidden;display:flex;flex-direction:column}
  #barre{padding:7px 10px;background:#1f1f1f;display:flex;flex-wrap:wrap;gap:6px;align-items:center;border-bottom:1px solid #333}
  button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:5px 9px;font-size:13px;cursor:pointer}
  button.actif{background:#c79a2b;color:#111;border-color:#c79a2b}
  #aide{margin-left:auto;color:#999;font-size:12px}
  #corps{flex:1;display:flex;min-height:0}
  #liste{width:178px;overflow-y:auto;background:#1a1a1a;border-right:1px solid #333;padding:6px}
  .vignette{margin:0 0 8px;padding:4px;border:2px solid transparent;border-radius:6px;cursor:pointer}
  .vignette img{width:100%;display:block;border-radius:3px;background:#000}
  .vignette span{display:block;font-size:11px;color:#aaa;margin-top:2px}
  .vignette.active{border-color:#c79a2b;background:#262626}
  .vignette.vue span::after{content:"  ✓ vue";color:#5ad16a}
  #vue{position:relative;flex:1;overflow:hidden;cursor:grab;background:#000}
  #vue img{position:absolute;top:0;left:0;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
  #regles{position:absolute;inset:0;pointer-events:none}
  #etiquette{position:absolute;left:34px;bottom:10px;background:rgba(20,20,20,.85);border:1px solid #c79a2b;color:#ffd978;padding:5px 9px;border-radius:6px;font-size:13px;font-variant-numeric:tabular-nums;pointer-events:none}
  #exemples{width:300px;overflow-y:auto;background:#1a1a1a;border-left:1px solid #333;padding:8px}
  #exemples h3{margin:0 0 6px;font-size:13px;color:#ffd978;font-weight:600}
  #exemples p{margin:0 0 8px;font-size:12px;color:#aaa}
  #exemples figure{margin:0 0 10px}
  #exemples img{width:100%;display:block;border-radius:3px;border:1px solid #444}
  #exemples figcaption{font-size:11.5px;color:#ff9c9c;margin-top:2px}
</style></head><body>
<div id="barre"><span id="aide">touches 1 à 5 : technique · ↑ ↓ : page · molette : zoom · glisser : déplacer · 0 : page entière</span></div>
<div id="corps">
  <div id="liste"></div>
  <div id="vue"><img id="img" alt=""><canvas id="regles"></canvas><div id="etiquette"></div></div>
  <div id="exemples"><h3>Ce qu'il faut chercher</h3><p>Exemples fabriqués, vus par la technique affichée. Un titre ferait deux ou trois lignes de lettres d'environ 4 mm de haut.</p><div id="ex"></div></div>
</div>
<script src="donnees.js"></script>
<script>
const barre=document.getElementById('barre'),liste=document.getElementById('liste'),vue=document.getElementById('vue'),img=document.getElementById('img'),
      cv=document.getElementById('regles'),etq=document.getElementById('etiquette'),ex=document.getElementById('ex'),aide=document.getElementById('aide');
let T=D.noms.length-1,P=0,z=1,tx=0,ty=0,cur=null,vues={};
try{vues=JSON.parse(localStorage.getItem('pages_vues')||'{}')}catch(e){}
const deux=n=>String(n).padStart(2,'0');
D.noms.forEach((n,i)=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>technique(i);barre.insertBefore(b,aide)});
D.pages.forEach((p,k)=>{const d=document.createElement('div');d.className='vignette';d.onclick=()=>page(k);
  d.innerHTML='<img loading="lazy" alt=""><span>page '+(k+1)+' · '+p.zone+' · x '+p.x.toFixed(0)+'–'+(p.x+D.larg).toFixed(0)+' · y '+p.y.toFixed(0)+'–'+(p.y+D.haut).toFixed(0)+'</span>';liste.appendChild(d)});
function technique(i){T=i;[...barre.querySelectorAll('button')].forEach((b,j)=>b.classList.toggle('actif',j===i));
  [...liste.children].forEach((d,k)=>d.querySelector('img').src='pages/t'+(T+1)+'_p'+deux(k+1)+'.jpg');
  ex.innerHTML=D.exemples[T].map((c,j)=>'<figure><img src="exemples/t'+(T+1)+'_e'+(j+1)+'.jpg" alt=""><figcaption>'+c+'</figcaption></figure>').join('');
  img.src='pages/t'+(T+1)+'_p'+deux(P+1)+'.jpg';dessiner()}
function page(k){P=Math.max(0,Math.min(D.pages.length-1,k));[...liste.children].forEach((d,j)=>d.classList.toggle('active',j===P));
  liste.children[P].scrollIntoView({block:'nearest'});vues[P]=1;try{localStorage.setItem('pages_vues',JSON.stringify(vues))}catch(e){}
  [...liste.children].forEach((d,j)=>d.classList.toggle('vue',!!vues[j]));img.src='pages/t'+(T+1)+'_p'+deux(P+1)+'.jpg';ajuster()}
function ajuster(){const r=vue.getBoundingClientRect(),w=D.larg/D.mm,h=D.haut/D.mm;z=Math.min((r.width-40)/w,(r.height-40)/h);tx=30+(r.width-30-w*z)/2;ty=24+(r.height-24-h*z)/2;poser()}
function poser(){img.style.transform='translate('+tx+'px,'+ty+'px) scale('+z+')';dessiner()}
function dessiner(){const r=vue.getBoundingClientRect(),dpr=window.devicePixelRatio||1;cv.width=r.width*dpr;cv.height=r.height*dpr;cv.style.width=r.width+'px';cv.style.height=r.height+'px';
  const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,r.width,r.height);const p=D.pages[P],pm=z/D.mm;
  g.fillStyle='rgba(20,20,20,.88)';g.fillRect(0,0,r.width,22);g.fillRect(0,0,30,r.height);g.strokeStyle='#ffc94a';g.fillStyle='#ffc94a';g.font='11px -apple-system,Helvetica,Arial';g.lineWidth=1;
  const pas=pm>40?1:(pm>12?5:10);
  for(let m=Math.ceil((p.x-(tx-30)/pm-40));m<=p.x+(r.width-tx)/pm;m++){const X=tx+(m-p.x)*pm;if(X<30||X>r.width)continue;const gd=m%pas===0;g.beginPath();g.moveTo(X,gd?8:16);g.lineTo(X,22);g.stroke();if(gd)g.fillText(m+' mm',X+3,11)}
  for(let m=Math.ceil((p.y-ty/pm-40));m<=p.y+(r.height-ty)/pm;m++){const Y=ty+(m-p.y)*pm;if(Y<22||Y>r.height)continue;const gd=m%pas===0;g.beginPath();g.moveTo(gd?14:22,Y);g.lineTo(30,Y);g.stroke();if(gd){g.save();g.translate(11,Y-3);g.rotate(-Math.PI/2);g.fillText(m+'',0,0);g.restore()}}
  etq.textContent=D.noms[T]+'  ·  page '+(P+1)+'/'+D.pages.length+' ('+D.pages[P].zone+')'+(cur?'  ·  curseur : x '+cur[0].toFixed(1)+' mm, y '+cur[1].toFixed(1)+' mm':'')}
addEventListener('keydown',e=>{const n=parseInt(e.key);if(n>=1&&n<=D.noms.length)technique(n-1);else if(e.key==='0')ajuster();
  else if(e.key==='ArrowDown'||e.key==='ArrowRight'){e.preventDefault();page(P+1)}else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){e.preventDefault();page(P-1)}});
vue.addEventListener('wheel',e=>{e.preventDefault();const r=vue.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.15:1/1.15;tx=mx-(mx-tx)*f;ty=my-(my-ty)*f;z*=f;poser()},{passive:false});
let d=null;vue.addEventListener('mousedown',e=>{d=[e.clientX-tx,e.clientY-ty];vue.style.cursor='grabbing'});
addEventListener('mouseup',()=>{d=null;vue.style.cursor='grab'});
addEventListener('mousemove',e=>{if(d){tx=e.clientX-d[0];ty=e.clientY-d[1];img.style.transform='translate('+tx+'px,'+ty+'px) scale('+z+')'}
  const r=vue.getBoundingClientRect(),p=D.pages[P];cur=[p.x+(e.clientX-r.left-tx)/z*D.mm,p.y+(e.clientY-r.top-ty)/z*D.mm];dessiner()});
addEventListener('resize',ajuster);
technique(T);page(0);
</script></body></html>
"""


if __name__ == "__main__":
    main()
