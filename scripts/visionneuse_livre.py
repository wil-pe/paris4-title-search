#!/usr/bin/env python3
"""Visionneuse « façon livre » de PHerc. Paris 4 : le papyrus déroulé, une colonne de texte par page.

Sources (tout est déjà sur le disque, rien n'est téléchargé) :
- les deux cartes d'encre du rouleau entier déroulé (segment spirale 20260926044226), 2390 × 142283 px à
  76,8 µm par pixel, soit 10,93 m × 184 mm ; le texte se lit de gauche à droite et finit à droite ;
- pour la fin du rouleau, les cartes bien plus nettes du segment w010-027 : student_propre / avril_propre
  (9,6 µm, x ≥ 172,8 mm) et la carte 2D « canonical » réduite à 19,2 µm (tout le segment) ;
- l'édition DCLP 989336 (16 dernières colonnes), affichée à côté de l'image.

Découpage en pages :
1. score d'entrecolonne le long du rouleau : par bandes horizontales, creux de densité d'encre large de
   3 mm comparé aux deux flancs (6 à 30 mm) ; la part qui se répète d'une spire à l'autre à la même phase
   (plis et déchirures du rouleau) est retirée ;
2. suivi par programmation dynamique d'une suite d'entrecolonnes espacés d'environ un pas de colonne ;
3. un entrecolonne est « confirmé » si son score dépasse SEUIL ; une colonne est dite détectée quand elle
   est bordée par deux entrecolonnes d'une suite d'au moins MIN_SUITE confirmés. Ailleurs, les pages
   suivent le découpage régulier du suivi (largeur ≈ pas local, léger recouvrement) sans prétendre
   tomber sur les vrais entrecolonnes.

Les trois dernières colonnes et le papyrus vierge qui suit sont montrés dans le repère du segment de fin
(x 0–408 mm, y 0–167 mm), celui de tous les autres outils du labo ; les cartes du rouleau entier y sont
ramenées par une transformation ajustée sur des blocs appariés.

  .venv/bin/python -I scripts/visionneuse_livre.py            # tout (≈ 3 min)
  .venv/bin/python -I scripts/visionneuse_livre.py --texte    # transcription, traduction et page HTML seulement
Les lectures automatiques des pages sans édition (scripts/lecture_livre.py, une sous-tâche par page) sont lues
dans apercus/livre/lectures/pNNN.json et affichées à droite, avec leurs lignes repérées sur l'image.

Sortie : apercus/livre/index.html (+ pages/, vignettes/, donnees.js), servie par le serveur « apercus » :
http://localhost:8765/livre/index.html
"""
import glob
import json
import pathlib
import sys
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates, uniform_filter1d
from scipy.signal import fftconvolve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
P4 = V.DATA / "PHercParis4"
SPIRALE = P4 / "segments/20260926044226-spiral-output-generation-19"
FIN = P4 / "segments/20260701183124-w010-027"
TEI = V.RACINE / "externe/idp.data/DCLP/989/989336.xml"
DOS = V.RACINE / "apercus/livre"

MM = 0.0768                # mm par pixel des cartes du rouleau entier (pas du maillage 192 µm / 2,5)
F = 4                      # réduction pour la détection
PX = MM * F                # 0,307 mm
NB = 14                    # bandes horizontales
SEUIL = 0.25               # score minimal d'un entrecolonne confirmé
MIN_SUITE = 4              # nombre minimal d'entrecolonnes confirmés à la suite
MARGE_G, MARGE_D = 6.0, 3.0   # mm montrés à gauche et à droite d'une colonne détectée
RECOUV = 5.0               # recouvrement de chaque côté des pages du découpage régulier
SEG_X, SEG_Y = 408.0, 167.4   # étendue du segment de fin, en mm
X_PROPRE = 172.8           # début des cartes « propres » à 9,6 µm dans ce segment
VERIFIEES = {1, 2, 3, 8, 15}  # colonnes de l'édition dont le texte a été relu sur l'image (voir apparier)

CARTES = {                 # identifiant -> (nom du bouton, nom complet dans l'étiquette)
    "s": ("fin · student 3D", "segment de fin, modèle 3D student sans fantômes, 19,2 µm/px"),
    "a": ("fin · avril 3D", "segment de fin, modèle 3D d'avril sans fantômes, 19,2 µm/px"),
    "c": ("fin · canonical 2D", "segment de fin, modèle 2D canonical, 19,2 µm/px"),
    "k": ("rouleau · student 3D", "rouleau entier, modèle 3D student-v12A, 76,8 µm/px"),
    "h": ("rouleau · hecate 2D", "rouleau entier, modèle 2D hecate, 76,8 µm/px"),
}

# Traduction de travail, en français, des colonnes éditées : {colonne: [(lignes de l'édition, texte)]}.
# Faite à partir du texte grec retenu par l'édition (sans les conjectures de l'apparat), elle n'a été relue
# par aucun spécialiste. […] = lacune ou passage trop mutilé ; (?) = sens incertain.
TRADUCTION = {
    16: [("1–12", "Il ne reste que des traces de lettres : rien de traduisible.")],
    15: [("1–2", "[…] à notre sujet."),
         ("2–6", "Et sont pareillement sans finesse ceux qui louent, en musique, ce qui saisit et ce qui émerveille […]"),
         ("9–11", "car ce qui saisit se développera en même temps (?) [et] se relâchera […]")],
    14: [("2", "[…] au plaisir —"),
         ("3–6", "non pas, certes, que cela le produise d'emblée, comme tel autre élément de la musique —"),
         ("6–7", "ou bien, en troisième lieu, supposer que certains […]"),
         ("9", "[…] la plupart […]")],
    13: [("1–4", "[…] il rendra l'éloge vide et vain."),
         ("4–10", "Ce qui tient à la durée (au fait de s'y attarder) pourrait donc peut-être se produire dans certains cas ; "
                  "mais ce qui tient à l'information reçue et à l'inférence que l'on en tire […]")],
    12: [("1–2", "[…] qui se produit par comparaison,"),
         ("2–10", "et (des gens) qui nous montrent par l'exemple, à propos de la pourpre et des choses semblables, que, "
                  "après les avoir tenues pour excellentes, lorsque nous les mettons côte à côte, nous reconnaissons "
                  "aussi que nous n'étions pas dans le vrai auparavant."),
         ("10–12", "D'abord, en effet, de la vue elle-même […]")],
    11: [("2–6", "[…] le mouvement (l'émotion) qu'il produit à présent, parce que celui-ci n'est pas forcément supérieur "
                 "à celui qui était alors le meilleur,"),
         ("6–8", "(et que) ce qui s'est imposé le premier fausse […]"),
         ("10–18", "[…] la supériorité […] chacun des deux mouvements (?) […] chacun est considéré […] nous […]")],
    10: [("3–9", "[…] de l'agrégat, pénétrant incidemment dans cet organe des sens par lequel nous portons nos jugements "
                 "sur ce genre de chose, produira la modification (?) des pores […]"),
         ("11–15", "[…] aux opinions antérieures […], mais l'emportant selon […] les critères."),
         ("15–18", "Car nous les supprimerions plutôt, bien loin de les admettre, si vraiment nous […]")],
    9: [("2–5", "[…] obscur […] bien saisir un sens de ce genre, mais, à partir du reste, pas le moindre."),
        ("6–7", "Ainsi […] entièrement […]"),
        ("11–19", "[…] des concours (?) […] ces propos, parce que ce qui, souvent, est digne d'éloge aux yeux de ces "
                  "gens-là mêmes, cela même est souvent aussi reconnu de tous […]")],
    8: [("1–5", "[…] propre […] de ce qu'il y a de tel chez Xénophante (nom en partie restitué), et que d'autres aussi "
                "passent pour produire ;"),
        ("5–9", "et pareillement, chez un autre non plus, rien qui soit propre à ce qui passe pour l'être (?) et qui "
                "plaise davantage aussi au plus grand nombre ;"),
        ("9–14", "mais, comme pour les aliments aussi, nous ne jugeons pas d'emblée que les choses rares soient forcément "
                 "plus agréables que les abondantes."),
        ("14–18", "Car […] ne serait pas […], et la câpre […] abondant […] à ceux-là, parce que rare […]")],
    7: [("2–4", "[…] de même que certains aliments aussi ont réellement une supériorité sur ceux qui abondent ;"),
        ("5–13", "mais on examinera souvent de telles choses de la façon suivante : lorsque le plus abondant est à "
                 "disposition, la nature s'en trouve-t-elle plus agréablement que de celui-ci (?) ; et, inversement, "
                 "celui-ci, si on le présente souvent, procure-t-il quelque plaisir ?"),
        ("14–18", "Non pas l'autre, assurément — de même que, pour les saveurs, souvent le pain d'orge (ne donne) par "
                  "lui-même un agrément en rien moindre […]")],
    6: [("1–6", "[…] examiner de près, afin que nous ne soyons pas abusés par le rare, comme s'il possédait une "
                "supériorité, ainsi que le sont la plupart des gens."),
        ("6–14", "Il arrive en effet parfois que, pour avoir été émus par la rareté elle-même, certains naïfs, croyant dès "
                 "lors que la chose elle-même possède supériorité et plaisir, soient attirés vers elle."),
        ("14–19", "Car il nous est arrivé d'être attirés alors que l'une des deux choses était rare et de moindre prix, "
                  "et l'autre abondante […]")],
    5: [("1–2", "[…] propre […] après avoir été attirés."),
        ("2–6", "Pourtant, sur la durée, nous ne persistons pas à nous y attacher plus qu'à l'autre."),
        ("6–10", "C'est pourquoi il ne faut pas condamner […] et par le fait de produire une supériorité […] ;"),
        ("11–18", "mais, plus encore que cela, […] du fait de la rareté nous y avons prêté attention, […] souvent […], "
                  "lorsque (les choses) sont dissemblables, c'est à cause de lui (le rare ?) que nous y avons adhéré, et […]")],
    4: [("1–3", "[…] et souvent quelqu'un ne nous a pas moins plu, mais même davantage."),
        ("4–8", "Et en louant d'après ce que nous avons éprouvé, nous ne nous trompons pas : car nous avons été fortement "
                "émus, à moins que quelque chose ne l'ait empêché,"),
        ("9–13", "de même que, pour les saveurs, nous déclarons […] celui qui reste en deçà ou qui, souvent, procure "
                 "le plaisir […]"),
        ("16–18", "[…] ce que nous dirons, précisément, être inférieur. De cette façon donc […]")],
    3: [("1–4", "[…] celui qui n'a pas de discernement, nous disons qu'il est, de ces choses, un juge sans règle fixe ;"),
        ("4–8", "car, de même que […] seul juge et calcule comment il faut apprécier les supériorités."),
        ("8–11", "Le musicien (?), lui, sur toutes choses […] sur celles-là."),
        ("11–18", "Sur le plaisir, en effet, nous céderons le pas […] juge(s) (?) […] et à certains, mais non sur celui "
                  "du goût ; à d'autres encore […]")],
    2: [("1–3", "[…] sont examinés au moyen des critères propres à chacune ;"),
        ("3–9", "et de plus, ils n'ont rien à dire ni du plaisir en général ni du plaisir particulier, rien non plus de "
                "défini (?), mais seulement des […] ;"),
        ("9–12", "c'est pourquoi ils se disputent aussi entre eux."),
        ("12–17", "Et s'il fallait céder le pas à quelqu'un, qui le céderait à celui qui ne juge de rien d'autre, ni par "
                  "rien d'autre, que par ce qu'il éprouve ?"),
        ("17–19", "Mais, pour avoir souvent entendu […]")],
    1: [("1–3", "[…] en passant, de juger certaines choses et d'en retenir d'autres ;"),
        ("3–6", "et puisse-t-il nous apparaître que nous disons vrai, comme cela pourrait souvent apparaître. "
                "(Fin du texte : le reste du rouleau est vierge.)")],
}


# ------------------------------------------------------------------ détection des colonnes

def charger():
    f = sorted(glob.glob(str(SPIRALE / "ink-detection/*.tif")))
    h = tifffile.imread(next(x for x in f if "hecate" in x))
    k = tifffile.imread(next(x for x in f if "ink3d" in x))
    return h, k


def par_bandes(a, seuil):
    """Nombre de pixels au-dessus du seuil, par bande horizontale et par pas de PX : tableau (NB, W/F)."""
    w = a.shape[1] // F
    b = np.linspace(0, a.shape[0], NB + 1).astype(int)
    return np.stack([(a[b[i]:b[i + 1], :w * F] > seuil).sum(0).reshape(w, F).sum(1) for i in range(NB)]).astype(np.float32)


def creux(encre, valide, plein):
    """Score d'entrecolonne en chaque x : profondeur relative du creux d'encre, moyennée sur les bandes
    où il y a du papyrus et de l'encre des deux côtés."""
    dens = np.where(valide / plein > 0.6, encre / np.maximum(valide, 1), np.nan)
    d0, ok = np.nan_to_num(dens), (~np.isnan(dens)).astype(np.float32)
    lis = lambda a, n: uniform_filter1d(a, n, axis=1, mode="constant")
    n3, n24, d18 = int(3 / PX), int(24 / PX), int(18 / PX)
    centre = lis(d0, n3) / np.maximum(lis(ok, n3), 1e-3)

    def flanc(sens):
        return np.roll(lis(d0, n24), sens * d18, 1) / np.maximum(np.roll(lis(ok, n24), sens * d18, 1), 1e-3), np.roll(lis(ok, n24), sens * d18, 1)
    (g, pg), (d, pd) = flanc(+1), flanc(-1)
    m = np.minimum(g, d)
    preuve = (pg > 0.7) & (pd > 0.7) & (lis(ok, n3) > 0.9) & (m > 0.01)
    v = np.where(preuve, np.clip(1 - centre / np.maximum(m, 1e-4), 0, 1), 0)
    nb = preuve.sum(0)
    return np.where(nb >= 4, v.sum(0) / np.maximum(nb, 1), 0).astype(np.float32)


def spires():
    """Abscisse (mm) du début de chaque spire dans l'image déroulée, de l'extérieur vers le cœur."""
    meta = json.loads(next((SPIRALE / "mesh").glob("*.tifxyz/meta.json")).read_text())
    return np.array(sorted(w["column_begin"] for w in meta["winding_bounds"])) * 0.192


def sans_plis(s, x):
    """Retire de s ce qui se répète à la même phase dans les trois spires voisines de chaque côté."""
    deb = spires(); circ = np.diff(deb); n = 400
    u = np.stack([np.interp(deb[i] + circ[i] * np.arange(n) / n, x, s) for i in range(len(circ))])
    gab = np.stack([np.median([u[j] for j in range(i - 3, i + 4) if j != i and 0 <= j < len(u)], axis=0) for i in range(len(u))])
    gab = gaussian_filter1d(gab, 4, axis=1, mode="wrap")
    poids = np.clip((circ - 76) / 8, 0, 1)[:, None]        # rien à retirer quand la spire fait environ un pas de colonne
    u = np.clip(u - poids * gab, 0, None)
    out = s.copy()
    for i in range(len(circ)):
        a, b = np.searchsorted(x, deb[i]), np.searchsorted(x, deb[i + 1])
        out[a:b] = np.interp((x[a:b] - deb[i]) / circ[i] * n, np.arange(n), u[i])
    return out


def score(h, k):
    valide = par_bandes(k, 0)
    plein = (np.diff(np.linspace(0, k.shape[0], NB + 1).astype(int)) * F)[:, None]
    x = np.arange(valide.shape[1]) * PX
    s = [sans_plis(gaussian_filter1d(creux(par_bandes(a, t), valide, plein), 3), x) for a, t in ((h, 96), (k, 128))]
    return 0.5 * (s[0] + s[1]), x


def pas_local(s, x):
    """Pas des colonnes le long du rouleau : pic d'autocorrélation du score par fenêtres de 700 mm."""
    win = int(700 / PX); cx, cp = [], []
    for x0 in range(0, len(s) - win // 2, win // 4):
        t = s[x0:x0 + win] - s[x0:x0 + win].mean()
        f = np.fft.rfft(t, 2 * len(t)); a = np.fft.irfft(f * np.conj(f))[:len(t)]; a /= a[0] + 1e-9
        l0, l1 = int(60 / PX), int(80 / PX); i = l0 + int(np.argmax(a[l0:l1]))
        if a[i] >= 0.2 and 61 < i * PX < 79:
            cx.append((x0 + win / 2) * PX); cp.append(i * PX)
    return np.interp(x, cx, gaussian_filter1d(np.array(cp), 1.0))


def suivre(s, pas, fin_texte):
    """Suite d'entrecolonnes maximisant le score, chaque pas restant proche du pas local. Renvoie leurs
    abscisses (mm) et leur score ; la suite s'arrête au début de la dernière colonne."""
    n = len(s); dd = np.arange(int(56 / PX), int(86 / PX) + 1)
    meilleur = np.full(n, -1e9); avant = np.full(n, -1, int)
    meilleur[:dd[-1]] = s[:dd[-1]]
    for j in range(dd[0], n):
        i = j - dd[dd <= j]
        c = meilleur[i] - 0.15 * ((dd[:len(i)] * PX - pas[j]) / 5.0) ** 2
        m = int(np.argmax(c))
        if c[m] + s[j] > meilleur[j]:
            meilleur[j], avant[j] = c[m] + s[j], i[m]
    j = n - dd[-1] + int(np.argmax(meilleur[n - dd[-1]:])); g = []
    while j >= 0:
        g.append(j); j = avant[j]
    g = np.array(g[::-1]); g = g[g * PX <= fin_texte - 30]   # la dernière colonne fait au moins 30 mm
    appui = s[g]
    for m in np.flatnonzero(appui >= SEUIL):                 # recentrage des entrecolonnes confirmés sur le maximum local (±4 mm)
        a = max(0, g[m] - int(4 / PX)); g[m] = a + int(np.argmax(s[a:g[m] + int(4 / PX) + 1]))
    return g * PX, appui


def fin_du_texte(h, k):
    """Fin de la dernière colonne : dernière abscisse où deux bandes au moins portent autant d'encre que le
    texte qui précède, comme presque partout dans les 15 mm d'avant (le papyrus vierge n'a que des grains isolés)."""
    d = uniform_filter1d(par_bandes(h, 96) / np.maximum(par_bandes(k, 0), 1), int(3 / PX), axis=1)
    niveau, i = np.median(d[:, d.shape[1] // 2:]), 0
    for _ in range(3):                                       # le niveau de référence est pris dans le mètre de texte qui précède
        present = (d > niveau).sum(0) >= 2
        dense = uniform_filter1d(present.astype(np.float32), int(15 / PX)) >= 0.7
        i = int(np.flatnonzero(present & np.roll(dense, int(7.5 / PX))).max())
        niveau = np.median(d[:, i - int(1100 / PX):i - int(100 / PX)])
    return (i + 1) * PX + 1.5


def decouper(g, appui, debut, fin_texte, fin):
    """Liste des pages dans le repère du rouleau entier : colonnes détectées, découpage régulier, vierge."""
    conf = appui >= SEUIL
    suite = np.zeros(len(g), bool); i = 0
    while i < len(g):
        j = i
        while j < len(g) and conf[j]:
            j += 1
        if j - i >= MIN_SUITE:
            suite[i:j] = True
        i = max(j, i + 1)
    bornes = [debut] + list(g) + [fin_texte]                 # le dernier entrecolonne ouvre la dernière colonne
    sur = [False] + list(suite) + [True]
    pages = []
    for a, b, sa, sb in zip(bornes[:-1], bornes[1:], sur[:-1], sur[1:]):
        if b - a < 25:                                       # reste trop étroit au tout début du rouleau
            continue
        pages.append(dict(type="col" if sa and sb else "reg", a=float(a), b=float(b)))
    n = max(1, round((fin - fin_texte) / 65))
    for i in range(n):
        pages.append(dict(type="vierge", a=float(fin_texte + (fin - fin_texte) * i / n), b=float(fin_texte + (fin - fin_texte) * (i + 1) / n)))
    cols = [p for p in pages if p["type"] != "vierge"]
    for i, p in enumerate(cols):
        p["fin"] = len(cols) - i                             # rang à partir de la fin, comme dans l'édition
    for i, p in enumerate(pages):
        p["num"] = i + 1
        g_, d_ = (MARGE_G, MARGE_D) if p["type"] == "col" else (RECOUV, RECOUV)
        p["x0"], p["x1"] = max(debut, p["a"] - g_), min(fin, p["b"] + d_)
    return pages


# ------------------------------------------------------------------ édition (TEI -> lignes)

NS = "{http://www.tei-c.org/ns/1.0}"
ENTOURE = {"surplus": ("{", "}"), "del": ("⟦", "⟧"), "add": ("\\", "/")}


def lacune(el):
    perdu, ligne = el.get("reason") == "lost", el.get("unit") == "line"
    q = el.get("quantity")
    if ligne:
        return "– – – – –" if q is None else f"(traces de {q} lignes)"
    if q is None and el.get("atLeast"):
        t = "." * int(el.get("atLeast")) + "(" + "." * (int(el.get("atMost")) - int(el.get("atLeast"))) + ")"
    elif q is None:
        t = "– – –"
    elif int(q) > 8 or el.get("precision") == "low":
        t = f"– ca. {q} –"
    else:
        t = "." * int(q)
    return f"[{t}]" if perdu else t


def edition():
    """{numéro de colonne: [ {n: numéro de ligne, t: texte, s: signe marginal sous la ligne} ]}, en notation de Leyde."""
    racine = ET.parse(TEI).getroot()
    ed = next(d for d in racine.iter(NS + "div") if d.get("type") == "edition")
    cols = {}
    for div in ed.findall(NS + "div"):
        lignes = [dict(n="", t="")]
        incertain = [0]

        def texte(s):
            s = s.replace("\n", "")
            if incertain[0]:
                s = "".join(c + "̣" if c.isalpha() else c for c in unicodedata.normalize("NFC", s))
            lignes[-1]["t"] += s

        def contenu(el):
            if el.text:
                texte(el.text)
            for e in el:
                rendre(e)
                if e.tail:
                    texte(e.tail)

        def rendre(el):
            tag = el.tag.replace(NS, "")
            if tag in ("head", "rdg", "reg", "desc", "certainty", "note"):
                return
            if tag == "lb":
                if el.get("break") == "no":
                    lignes[-1]["t"] = lignes[-1]["t"].rstrip() + "-"
                lignes.append(dict(n=el.get("n"), t=""))
            elif tag == "gap":
                if el.get("unit") == "line":
                    q = el.get("quantity")
                    lignes[-1]["n"] = "" if q is None else f"{lignes[-1]['n']}–{int(lignes[-1]['n']) + int(q) - 1}"
                texte(lacune(el))
            elif tag == "milestone":
                lignes[-1]["s"] = el.get("rend")
            elif tag == "g":
                texte("※")
            elif tag == "unclear":
                incertain[0] += 1; contenu(el); incertain[0] -= 1
            elif tag == "supplied":
                o, f = ("⟨", "⟩") if el.get("reason") == "omitted" else ("[", "]")
                texte(o); contenu(el); texte(("?" if el.get("cert") == "low" else "") + f)
            else:
                o, f = ENTOURE.get(tag, ("", ""))
                texte(o); contenu(el); texte(f)

        for ab in div.findall(NS + "ab"):
            contenu(ab)
        for l in lignes:
            l["t"] = unicodedata.normalize("NFC", " ".join(l["t"].replace("][", "").split()))
        cols[int(div.get("n"))] = [l for l in lignes if l["t"]]
    return cols


def apparier(pages, ed):
    """Colonne n de l'édition = n-ième colonne à partir de la fin du rouleau.

    Vérifié en relisant le texte sur l'image pour les colonnes 1, 2, 3, 8 et 15 (« μὲν κρίνειν, τὰ δὲ »,
    « ἑκάστης κριτηρίων », « φρονοῦντα κριτὴν ἀ- », « γείνεσθαι, παραπλη- », « οἱ δὲ ὁμοίως καὶ οἱ τὸ »). Les autres suivent par
    comptage : on n'apparie donc que tant que la suite de colonnes détectées est ininterrompue depuis la fin."""
    for p in reversed([p for p in pages if p["type"] != "vierge"]):
        if p["type"] != "col" or p["fin"] not in ed:
            break
        p["ed"] = p["fin"]
        p["verif"] = "texte relu sur l'image" if p["fin"] in VERIFIEES else "par comptage depuis la fin (colonnes détectées sans interruption)"


# ------------------------------------------------------------------ repère du segment de fin

def reduire(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return np.asarray(a[:h, :w]).reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def correlation(img, gab):
    img = img - img.mean(); gab = gab - gab.mean()
    c = fftconvolve(img, gab[::-1, ::-1], mode="valid")
    n = np.sqrt(fftconvolve(img ** 2, np.ones_like(gab), mode="valid").clip(1e-6)) * np.sqrt((gab ** 2).sum())
    return c / n


class Transfo:
    """Passage du repère du segment de fin (pixels de 76,8 µm : ligne r, colonne c) à celui du rouleau entier.

    Une translation les superpose à 3 mm près ; le reste (étirements différents des deux aplatissements) est
    décrit par un polynôme de degré 2 ajusté sur des blocs de 25 mm appariés par corrélation."""
    YB, XB, B, M = 165, 136917, 320, 70

    def __init__(self, h, k, canon, student):
        pts = []
        for ref, dc in ((reduire(canon, 4).astype(np.float32), 0), (student, int(round(X_PROPRE / MM)))):
            for r0 in range(40, ref.shape[0] - self.B - 20, 200):
                for c0 in range(0, ref.shape[1] - self.B, 200):
                    gab = ref[r0:r0 + self.B, c0:c0 + self.B]
                    ya, xa = self.YB + r0 - self.M, self.XB + dc + c0 - self.M
                    if gab.std() < 6 or ya < 0 or xa + self.B + 2 * self.M > k.shape[1] or ya + self.B + 2 * self.M > k.shape[0]:
                        continue
                    for a in (k, h):
                        c = correlation(a[ya:ya + self.B + 2 * self.M, xa:xa + self.B + 2 * self.M].astype(np.float32), gab)
                        i, j = np.unravel_index(np.argmax(c), c.shape); pic = c[i, j]
                        c[max(0, i - 8):i + 9, max(0, j - 8):j + 9] = -1
                        if pic > 0.15 and pic - c.max() > 0.06 and max(abs(i - self.M), abs(j - self.M)) < self.M - 2:
                            pts.append((r0 + self.B / 2, dc + c0 + self.B / 2, i - self.M, j - self.M))
        r, c, dy, dx = np.array(pts).T
        A = self.base(r, c)
        self.coef = []
        for v in (dy, dx):
            w = np.ones(len(v))
            for _ in range(8):                               # moindres carrés repondérés (valeurs aberrantes)
                co = np.linalg.lstsq(A * w[:, None], v * w, rcond=None)[0]; w = 1 / (1 + ((v - A @ co) / 2.5) ** 2)
            self.coef.append(co)
        self.residu = [float(np.median(np.abs(v - A @ co))) * MM for v, co in zip((dy, dx), self.coef)]
        self.blocs = len(pts)

    @staticmethod
    def base(r, c):
        r, c = np.asarray(r, float) / 1000, np.asarray(c, float) / 1000
        return np.stack([np.ones_like(r), r, c, r * r, c * c, r * c], -1)

    def vers_rouleau(self, r, c):
        b = self.base(r, c)
        return self.YB + r + b @ self.coef[0], self.XB + c + b @ self.coef[1]

    def x_segment(self, x_rouleau, y=SEG_Y / 2):
        """Abscisse (mm) dans le segment de fin du point d'abscisse x_rouleau (mm), à mi-hauteur."""
        r, c = y / MM, x_rouleau / MM - self.XB
        for _ in range(8):
            c = x_rouleau / MM - self.XB - float(self.base(r, c) @ self.coef[1])
        return c * MM

    def x_rouleau(self, x_segment, y=SEG_Y / 2):
        return float(self.vers_rouleau(y / MM, x_segment / MM)[1]) * MM

    def ramener(self, a, x0, x1, pas=0.0384):
        """Carte du rouleau entier rééchantillonnée dans le repère du segment, pour x0 ≤ x < x1 mm."""
        y, x = np.mgrid[0:int(SEG_Y / pas), 0:int((x1 - x0) / pas)].astype(np.float32)
        r, c = self.vers_rouleau(y * pas / MM, (x0 + x * pas) / MM)
        return map_coordinates(a, [r, c], order=1, mode="constant", cval=0).astype(np.uint8)


# ------------------------------------------------------------------ images

def jpeg(a, chemin, q=82):
    Image.fromarray(np.ascontiguousarray(a)).save(chemin, quality=q)


def vignette(a, chemin):
    h = 130 * a.shape[0] // a.shape[1]
    Image.fromarray(np.ascontiguousarray(a)).resize((130, min(h, 300)), Image.BILINEAR).save(chemin, quality=70)


def images_rouleau(p, h, k):
    c0, c1 = int(round(p["x0"] / MM)), int(round(p["x1"] / MM))
    lignes = np.flatnonzero((k[:, c0:c1] > 0).mean(1) > 0.03)
    r0, r1 = max(0, int(lignes.min()) - 20), min(k.shape[0], int(lignes.max()) + 21)
    p.update(repere="rouleau", x0=c0 * MM, y0=r0 * MM, w=(c1 - c0) * MM, h=(r1 - r0) * MM, cartes=["k", "h"], xr=[p["a"], p["b"]])
    for c, a in (("k", k), ("h", h)):
        jpeg(a[r0:r1, c0:c1], DOS / "pages" / f"p{p['num']:03d}_{c}.jpg")
    vignette(k[r0:r1, c0:c1], DOS / "vignettes" / f"p{p['num']:03d}.jpg")


def images_fin(p, h, k, T, canon, propres):
    """Page de la fin du rouleau, dans le repère du segment w010-027 : cartes nettes à 19,2 µm."""
    xr = [p["a"], p["b"]]
    p["a"], p["b"] = T.x_segment(xr[0]), min(SEG_X, T.x_segment(xr[1]))
    g, d = (MARGE_G, MARGE_D) if p["type"] == "col" else (RECOUV, RECOUV)
    c0, c1 = int(max(0, p["a"] - g) / 0.0192), int(min(SEG_X, p["b"] + d) / 0.0192)
    p.update(repere="segment", x0=c0 * 0.0192, y0=0.0, w=(c1 - c0) * 0.0192, h=canon.shape[0] * 0.0192, cartes=[], xr=xr)
    d0 = int(round(X_PROPRE / 0.0096))                       # première colonne des cartes propres, en pixels de 9,6 µm
    for c, a in propres.items():
        if c1 * 2 - d0 < 0.3 * (c1 - c0) * 2:                # moins de 30 % de la page couverte : inutile
            continue
        img = np.zeros((a.shape[0] // 2, c1 - c0), np.uint8)
        debut = max(0, c0 * 2 - d0)
        bloc = reduire(a[:, debut:c1 * 2 - d0], 2)
        img[:, img.shape[1] - bloc.shape[1]:] = bloc
        jpeg(img, DOS / "pages" / f"p{p['num']:03d}_{c}.jpg", 85); p["cartes"].append(c)
        if c == "s":
            entiere = c0 * 2 >= d0
    jpeg(canon[:, c0:c1], DOS / "pages" / f"p{p['num']:03d}_c.jpg", 85)
    if p["cartes"] and entiere:
        p["cartes"].append("c")
    else:
        p["cartes"].insert(0, "c")                           # carte par défaut = la première qui couvre toute la page
    for c, a in (("k", k), ("h", h)):
        jpeg(T.ramener(a, p["x0"], p["x0"] + p["w"]), DOS / "pages" / f"p{p['num']:03d}_{c}.jpg"); p["cartes"].append(c)
    vignette(canon[:, c0:c1][::4, ::4], DOS / "vignettes" / f"p{p['num']:03d}.jpg")


def main():
    if "--texte" in sys.argv:                                # transcription, traduction et page HTML seulement
        ecrire(json.loads((DOS / "donnees.js").read_text(encoding="utf-8")[len("const D = "):].rstrip().rstrip(";")))
        return
    for d in ("pages", "vignettes"):
        (DOS / d).mkdir(parents=True, exist_ok=True)
    h, k = charger()
    print("cartes du rouleau entier :", h.shape, f"-> {h.shape[1] * MM / 1000:.2f} m × {h.shape[0] * MM:.0f} mm", flush=True)
    s, x = score(h, k)
    fin_texte = fin_du_texte(h, k)
    g, appui = suivre(s, pas_local(s, x), fin_texte)
    cols = np.flatnonzero((k > 0).any(0)); debut, fin = cols.min() * MM, (cols.max() + 1) * MM
    pages = decouper(g, appui, debut, fin_texte, fin)
    ed = edition()
    apparier(pages, ed)
    nb = {t: sum(p["type"] == t for p in pages) for t in ("col", "reg", "vierge")}
    det = [p for p in pages if p["type"] == "col"]
    print(f"{len(pages)} pages : {nb['col']} colonnes détectées (x {det[0]['a']:.0f} à {det[-1]['b']:.0f} mm), "
          f"{nb['reg']} pages de découpage régulier, {nb['vierge']} pages de papyrus vierge ; fin du texte à {fin_texte:.0f} mm", flush=True)

    canon = np.asarray(Image.open(glob.glob(str(FIN / "ink-detection/downsampled/*new_canon*ds8.jpg"))[0]).convert("L"))
    propres = {c: np.load(P4 / "carte" / f, mmap_mode="r") for c, f in (("s", "student_propre.npy"), ("a", "avril_propre.npy"))}
    T = Transfo(h, k, canon, reduire(propres["s"], 8).astype(np.float32))
    print(f"repère du segment de fin : {T.blocs} blocs appariés, résidu médian {T.residu[1]:.2f} mm en x et {T.residu[0]:.2f} mm en y ; "
          f"dernière colonne à x = {T.x_segment(pages[-nb['vierge'] - 1]['a']):.0f}–{T.x_segment(fin_texte):.0f} mm du segment", flush=True)
    for p in pages:
        if p["a"] > T.XB * MM and T.x_segment(p["a"]) - MARGE_G >= 0:   # page entièrement dans le segment de fin
            images_fin(p, h, k, T, canon, propres)
        else:
            images_rouleau(p, h, k)
        for c in ("a", "b", "x0", "y0", "w", "h"):
            p[c] = round(p[c], 2)
        p["xr"] = [round(v, 1) for v in p["xr"]]
        if p["num"] % 20 == 0:
            print("  page", p["num"], flush=True)
    ecrire(dict(pages=pages, nb=nb, seuil=SEUIL))


def ecrire(donnees):
    """Écrit donnees.js (pages + cartes, transcription et traduction à jour) et la page HTML."""
    lectures = {int(f.stem[1:]): json.loads(f.read_text(encoding="utf-8")) for f in sorted((DOS / "lectures").glob("p[0-9]*.json"))}
    temoins = DOS / "lectures" / "temoins.json"               # fiabilité mesurée sur des colonnes déjà éditées (lecture_livre.py)
    donnees.update(lectures=lectures, temoins=json.loads(temoins.read_text(encoding="utf-8")) if temoins.exists() else [])
    donnees.update(cartes=CARTES, edition=edition(), traduction=TRADUCTION,
                   source="édition : J. Kern, rév. H. Essler, DCLP/TM 989336 (papyri.info), CC BY 3.0")
    (DOS / "donnees.js").write_text("const D = " + json.dumps(donnees, ensure_ascii=False) + ";\n", encoding="utf-8")
    (DOS / "index.html").write_text(HTML, encoding="utf-8")
    print("visionneuse écrite :", DOS / "index.html")


HTML = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>PHerc. Paris 4 — lecture colonne par colonne</title>
<style>
  html,body{height:100%}
  body{margin:0;background:#141414;color:#e8e8e8;font:14px/1.35 -apple-system,Helvetica,Arial,sans-serif;overflow:hidden;display:flex;flex-direction:column}
  #barre{padding:7px 10px;background:#1f1f1f;display:flex;flex-wrap:wrap;gap:6px;align-items:center;border-bottom:1px solid #333}
  button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:5px 9px;font-size:13px;cursor:pointer}
  button.actif{background:#c79a2b;color:#111;border-color:#c79a2b}
  select{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:5px 6px;font-size:13px;max-width:210px}
  #pos{min-width:92px;text-align:center;font-variant-numeric:tabular-nums}
  .sep{width:1px;height:22px;background:#444;margin:0 4px}
  #barre label{font-size:12px;color:#aaa;display:flex;align-items:center;gap:4px}
  #aide{flex-basis:100%;color:#999;font-size:12px}
  #corps{flex:1;display:flex;min-height:0}
  #liste{width:158px;overflow-y:auto;background:#1a1a1a;border-right:1px solid #333;padding:6px}
  .vignette{margin:0 0 8px;padding:4px;border:2px solid transparent;border-radius:6px;cursor:pointer}
  .vignette img{width:100%;display:block;border-radius:3px;background:#000;min-height:60px}
  .vignette span{display:block;font-size:11px;color:#aaa;margin-top:2px}
  .vignette.active{border-color:#c79a2b;background:#262626}
  .vignette b{color:#ffd978;font-weight:600}
  .vignette i{font-style:normal;color:#7fc6ff}
  #vue{position:relative;flex:1;overflow:hidden;cursor:grab;background:#000}
  #vue img{position:absolute;top:0;left:0;transform-origin:0 0;user-select:none;-webkit-user-drag:none;max-width:none}
  #regles{position:absolute;inset:0;pointer-events:none}
  #etiquette{position:absolute;left:34px;bottom:10px;max-width:calc(100% - 60px);background:rgba(20,20,20,.88);border:1px solid #c79a2b;color:#ffd978;padding:5px 9px;border-radius:6px;font-size:13px;font-variant-numeric:tabular-nums;pointer-events:none}
  #attente{position:absolute;right:12px;top:30px;background:rgba(20,20,20,.85);color:#aaa;padding:3px 8px;border-radius:5px;font-size:12px;display:none}
  #texte{width:340px;overflow-y:auto;background:#1a1a1a;border-left:1px solid #333;padding:10px 12px}
  #texte.cache{display:none}
  #texte h3{margin:0 0 4px;font-size:14px;color:#ffd978;font-weight:600}
  #texte p{margin:0 0 10px;font-size:12px;color:#aaa}
  .l{display:flex;gap:8px;align-items:baseline;padding:1px 0}
  .l .n{width:34px;flex:none;text-align:right;color:#777;font-size:11px;font-variant-numeric:tabular-nums}
  .l .g{font:17px/1.4 "New Athena Unicode","GFS Didot","Gentium Plus","Times New Roman",serif;color:#f1ece0}
  .maj .l .g{letter-spacing:.06em;font-size:16px}
  .signe{margin:-2px 0 0 42px;font-size:11px;color:#7fc6ff}
  #tr h4{margin:16px 0 4px;font-size:13px;color:#ffd978;font-weight:600;border-top:1px solid #333;padding-top:10px}
  .f{display:flex;gap:8px;align-items:baseline;padding:3px 0 7px}
  .f .n{width:34px;flex:none;text-align:right;color:#5f8f86;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap}
  .f .t{font:italic 14px/1.4 Georgia,"Times New Roman",serif;color:#9fd3c7}
  .lec{cursor:pointer;border-left:3px solid #555;padding-left:5px}
  .lec.c3{border-color:#5ad16a}.lec.c2{border-color:#e6c84a}.lec.c1{border-color:#e08a3c}
  .lec .g{font:16px/1.35 "New Athena Unicode","GFS Didot","Times New Roman",serif;letter-spacing:.05em;word-break:break-all}
  .lec .g u{text-decoration:none;color:#8f8a7c}
  .lec:hover{background:#262626}
  #lm{margin:8px 0;font:italic 14px/1.4 Georgia,serif;color:#9fd3c7}
  .signe::before{content:"";display:inline-block;width:26px;border-top:2px solid #7fc6ff;vertical-align:middle;margin-right:6px}
</style></head><body>
<div id="barre">
  <button id="prec" title="page précédente (←)">◀</button><span id="pos"></span><button id="suiv" title="page suivante (→)">▶</button>
  <span class="sep"></span><span id="cartes"></span><span class="sep"></span>
  <button id="b-debut">début du rouleau</button><button id="b-ed">1re colonne éditée</button><button id="b-fin">dernière colonne</button>
  <select id="mieux" title="pages sans édition classées par nombre de lettres lues avec assurance"><option value="">pages les mieux lues…</option></select>
  <span class="sep"></span><label>lumière <input id="gain" type="range" min="1" max="4" step="0.1" value="1"></label>
  <span id="aide">← → : tourner la page · ↑ ↓ : monter, descendre · molette : zoom · glisser : déplacer · c ou 1 à 5 : changer de carte d'encre · l : pleine largeur · 0 : page entière · b : bornes de la colonne · t : transcription · m : majuscules · f : traduction · r : lignes lues</span>
</div>
<div id="corps">
  <div id="liste"></div>
  <div id="vue"><img id="img" alt=""><canvas id="regles"></canvas><div id="etiquette"></div><div id="attente">chargement…</div></div>
  <div id="texte"><h3 id="tt"></h3><p id="tn"></p><div id="tl"></div><div id="tr"></div></div>
</div>
<script src="donnees.js"></script>
<script>
const $=id=>document.getElementById(id);
const vue=$('vue'),img=$('img'),cv=$('regles'),etq=$('etiquette'),liste=$('liste'),zc=$('cartes');
const N=D.pages.length,trois=n=>String(n).padStart(3,'0'),mil=v=>Math.round(v).toLocaleString('fr-FR');
let P=0,C=null,pref={},z=5,tx=0,ty=0,cur=null,mode='l',bornes=true,maj=false,gliss=null,trad=1,reperes=true;
const fichier=(p,c)=>'pages/p'+trois(p.num)+'_'+c+'.jpg';
const sures=l=>(l.lignes.map(x=>x.lu).join('').match(/[Α-Ω]/g)||[]).length;
function nom(p){return p.type==='vierge'?'papyrus vierge':(p.type==='col'?'':'≈ ')+'col. −'+p.fin}
function titre(p){
  if(p.type==='vierge')return 'papyrus vierge après la dernière colonne';
  if(p.type==='reg')return '≈ colonne −'+p.fin+' · découpage régulier, entrecolonnes non confirmés';
  return 'colonne −'+p.fin+(p.ed?' = col. '+p.ed+' de l’édition':'')+' · entrecolonnes détectés'}
D.pages.forEach((p,k)=>{const d=document.createElement('div');d.className='vignette';d.onclick=()=>page(k);
  d.innerHTML='<img loading="lazy" alt="" style="aspect-ratio:130/'+Math.min(130*p.h/p.w,300).toFixed(0)+'" src="vignettes/p'+trois(p.num)+'.jpg"><span>p. '+p.num+' · '+(p.type==='col'?'<b>'+nom(p)+'</b>':nom(p))+(p.ed?' <i>éd.</i>':'')+(D.lectures[p.num]&&!p.ed?' <i>lu·'+sures(D.lectures[p.num])+'</i>':'')+'</span>';liste.appendChild(d)});
const choix=q=>q.cartes.includes(pref[q.repere])?pref[q.repere]:q.cartes[0];
function carte(c,retenir){const p=D.pages[P];if(!p.cartes.includes(c))return;C=c;if(retenir)pref[p.repere]=c;
  [...zc.children].forEach(b=>b.classList.toggle('actif',b.dataset.c===C));$('attente').style.display='block';img.src=fichier(p,C);dessiner()}
img.onload=()=>{$('attente').style.display='none'};
function page(k,garder){P=Math.max(0,Math.min(N-1,k));const p=D.pages[P];
  [...liste.children].forEach((d,j)=>d.classList.toggle('active',j===P));liste.children[P].scrollIntoView({block:'nearest'});
  zc.innerHTML='';p.cartes.forEach((c,i)=>{const b=document.createElement('button');b.dataset.c=c;b.textContent=(i+1)+' · '+D.cartes[c][0];b.title=D.cartes[c][1];b.onclick=()=>carte(c,true);zc.appendChild(b)});
  $('pos').textContent='page '+p.num+' / '+N;cur=null;transcription();
  if(!garder)ajuster();carte(choix(p));
  [P-1,P+1].forEach(j=>{if(j>=0&&j<N){const q=D.pages[j];new Image().src=fichier(q,choix(q))}});
  try{localStorage.setItem('livre_page',p.num)}catch(e){} history.replaceState(null,'','#p'+p.num)}
function transcription(){const p=D.pages[P],e=p.ed?D.edition[p.ed]:null,tl=$('tl');
  $('tt').textContent=e?'Col. −'+p.ed+' de l’édition':(D.lectures[p.num]?'Pas d’édition publiée':'Pas de transcription');
  $('tn').textContent=e?'Appariement : '+p.verif+'. '+D.source+'. Lettres pointées : lecture incertaine ; [ ] : lacune ; points : lettres illisibles.'
    :(p.type==='vierge'?'Papyrus sans texte.':'L’édition publiée ne couvre que les 16 dernières colonnes du rouleau'+(p.type==='reg'?', et cette page vient du découpage régulier : elle peut être à cheval sur deux colonnes.':'.'));
  const f=p.ed?D.traduction[p.ed]:null,seg=s=>'<div class="f"><span class="n">l. '+s[0]+'</span><span class="t">'+s[1]+'</span></div>',apres={},places=new Set();
  if(f&&trad===1)f.forEach(s=>{const d=s[0].split('–').pop();(apres[d]=apres[d]||[]).push(s)});   // chaque passage traduit vient après sa dernière ligne
  let h=e?e.map(l=>'<div class="l"><span class="n">'+l.n+'</span><span class="g">'+(maj?majuscules(l.t):l.t)+'</span></div>'+(l.s?'<div class="signe">'+l.s+'</div>':'')
    +(apres[l.n]||[]).map(s=>{places.add(s);return seg(s)}).join('')).join(''):'';
  if(f&&trad===1)h+=f.filter(s=>!places.has(s)).map(seg).join('');
  tl.className=maj?'maj':'';tl.innerHTML=h;
  const lec=D.lectures[p.num],pointe=s=>s.replace(/[α-ω]/g,c=>'<u>'+c.toUpperCase()+'\u0323</u>'),tem=D.temoins.length?' Sur '+D.temoins.length+' colonnes témoins déjà éditées, lues de la même façon sans connaître l\u2019édition : '
      +D.temoins.map(x=>'col. −'+x.colonne+' : '+Math.round(100*x.retrouvees)+' % des lettres de l\u2019édition retrouvées, '+Math.round(100*x.justes_parmi_sures)+' % des lettres données pour sûres sont justes').join(' ; ')+'.':'';
  $('tr').innerHTML=(f?'<h4>Traduction de travail'+(trad?'':' (masquée)')+'</h4><p>Touche f : intercalée dans le grec, en bloc, ou masquée. Faite par Claude d\u2019après le texte grec de l\u2019édition (lecture retenue, sans les conjectures de l\u2019apparat) ; relue par aucun spécialiste, à vérifier avant tout usage. […] : lacune ; (?) : sens incertain.</p>'
    +(trad===2?f.map(seg).join(''):''):'')
    +(lec?'<h4>Lecture automatique à l\u2019aveugle'+(e?' (colonne témoin)':'')+'</h4><p>Faite par une sous-tâche de lecture à partir des seules cartes d\u2019encre : NON vérifiée, à prendre comme une piste et non comme une transcription. Lettre grisée et pointée : incertaine ; point : lettre illisible ; trait de couleur : confiance de la ligne (vert, jaune, orange, gris). Cliquer une ligne pour y aller.'+tem+'</p>'
      +(lec.remarques?'<p>'+lec.remarques+'</p>':'')
      +lec.lignes.map((l,i)=>'<div class="l lec c'+l.confiance+'" data-i="'+i+'"><span class="n">y '+Math.round(l.y)+'</span><span class="g">'+pointe(l.lu)+'</span></div>').join('')
      +(lec.mots&&lec.mots.length?'<div id="lm">Mots reconnus : '+lec.mots.map(m=>m.mot+(m.sur?'':' (?)')).join(', ')+'</div>':''):'');
  [...document.querySelectorAll('.lec')].forEach(d=>d.onclick=()=>{const l=lec.lignes[+d.dataset.i],r=vue.getBoundingClientRect();
    z=Math.max(z,(r.width-46)/Math.max(l.x1-l.x0+16,40));tx=38-(l.x0-8-p.x0)*z;ty=r.height/2-(l.y-p.y0)*z;poser()})}
function majuscules(t){return t.normalize('NFD').replace(/ͅ/g,'ι').replace(/[̀-̢̤-ͯ]/g,'').toUpperCase().replace(/Σ/g,'Ϲ').normalize('NFC')}
function ajuster(m){if(m)mode=m;const p=D.pages[P],r=vue.getBoundingClientRect(),W=r.width-30,H=r.height-22;
  if(mode==='0'){z=Math.min((W-16)/p.w,(H-16)/p.h);tx=30+(W-p.w*z)/2;ty=22+(H-p.h*z)/2}else{z=(W-16)/p.w;tx=38;ty=30}
  poser()}
function poser(){const p=D.pages[P];img.style.width=(p.w*z)+'px';img.style.height=(p.h*z)+'px';img.style.transform='translate('+tx+'px,'+ty+'px)';dessiner()}
function dessiner(){const p=D.pages[P],r=vue.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  if(cv.width!==Math.round(r.width*dpr)||cv.height!==Math.round(r.height*dpr)){cv.width=Math.round(r.width*dpr);cv.height=Math.round(r.height*dpr);cv.style.width=r.width+'px';cv.style.height=r.height+'px'}
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,r.width,r.height);
  if(bornes&&p.type!=='vierge'){g.strokeStyle=p.type==='col'?'rgba(90,209,106,.55)':'rgba(255,120,120,.45)';g.setLineDash([6,6]);g.lineWidth=1;
    [p.a,p.b].forEach(v=>{const X=tx+(v-p.x0)*z;g.beginPath();g.moveTo(X,22);g.lineTo(X,r.height);g.stroke()});g.setLineDash([])}
  const lec=D.lectures[p.num];if(reperes&&lec){const coul=['#888','#e08a3c','#e6c84a','#5ad16a'];g.lineWidth=2;lec.lignes.forEach(l=>{const Y=ty+(l.y+2.4-p.y0)*z;g.strokeStyle=coul[l.confiance]||'#888';g.globalAlpha=.75;g.beginPath();g.moveTo(tx+(l.x0-p.x0)*z,Y);g.lineTo(tx+(l.x1-p.x0)*z,Y);g.stroke()});
    g.globalAlpha=.6;g.strokeStyle='#7fc6ff';g.setLineDash([3,5]);g.lineWidth=1;(lec.entrecolonnes_x||[]).forEach(v=>{const X=tx+(v-p.x0)*z;g.beginPath();g.moveTo(X,22);g.lineTo(X,r.height);g.stroke()});g.setLineDash([]);g.globalAlpha=1}
  g.fillStyle='rgba(20,20,20,.88)';g.fillRect(0,0,r.width,22);g.fillRect(0,0,30,r.height);g.strokeStyle='#ffc94a';g.fillStyle='#ffc94a';g.font='11px -apple-system,Helvetica,Arial';g.lineWidth=1;
  const choisir=min=>[1,5,10,50,100,500].find(s=>s*z>=min)||1000,pasX=choisir(70),pasY=choisir(26),fin=choisir(4);
  for(let m=Math.ceil((p.x0-(tx-30)/z)/fin)*fin;m<=p.x0+(r.width-tx)/z;m+=fin){const X=tx+(m-p.x0)*z;if(X<30)continue;const gd=m%pasX===0;g.beginPath();g.moveTo(X,gd?8:16);g.lineTo(X,22);g.stroke();if(gd)g.fillText(mil(m)+' mm',X+3,11)}
  for(let m=Math.ceil((p.y0-(ty-22)/z)/fin)*fin;m<=p.y0+(r.height-ty)/z;m+=fin){const Y=ty+(m-p.y0)*z;if(Y<22)continue;const gd=m%pasY===0;g.beginPath();g.moveTo(gd?14:22,Y);g.lineTo(30,Y);g.stroke();if(gd){g.save();g.translate(11,Y-3);g.rotate(-Math.PI/2);g.fillText(m+'',0,0);g.restore()}}
  const rep=p.repere==='segment'?'repère du segment de fin, x '+p.a.toFixed(0)+'–'+p.b.toFixed(0)+' mm (≈ '+mil(p.xr[0])+'–'+mil(p.xr[1])+' mm du rouleau entier)':'repère du rouleau entier, x '+mil(p.a)+'–'+mil(p.b)+' mm';
  etq.textContent='PHerc. Paris 4 · page '+p.num+'/'+N+' · '+titre(p)+' · '+rep+' · carte : '+(C?D.cartes[C][1]:'')+(p.repere==='segment'&&(C==='k'||C==='h')?', recalée sur le segment':'')+(cur?' · curseur : x '+cur[0].toFixed(1)+' mm, y '+cur[1].toFixed(1)+' mm':'')}
addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey)return;const n=parseInt(e.key),p=D.pages[P],r=vue.getBoundingClientRect();
  if(n>=1&&n<=p.cartes.length)carte(p.cartes[n-1],true);
  else if(e.key==='c'||e.key==='C')carte(p.cartes[(p.cartes.indexOf(C)+1)%p.cartes.length],true);
  else if(e.key==='0')ajuster('0');else if(e.key==='l'||e.key==='L')ajuster('l');
  else if(e.key==='b'||e.key==='B'){bornes=!bornes;dessiner()}
  else if(e.key==='r'||e.key==='R'){reperes=!reperes;dessiner()}
  else if(e.key==='t'||e.key==='T'){$('texte').classList.toggle('cache');ajuster()}
  else if(e.key==='m'||e.key==='M'){maj=!maj;transcription()}
  else if(e.key==='f'||e.key==='F'){trad=(trad+1)%3;try{localStorage.setItem('livre_trad',trad)}catch(x){}transcription()}
  else if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();page(P+1)}else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();page(P-1)}
  else if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();const s=e.key==='ArrowDown'?-1:1;ty=Math.min(30,Math.max(r.height-40-p.h*z,ty+s*0.6*(r.height-22)));poser()}
  else if(e.key==='Home')page(0);else if(e.key==='End')page(N-1)});
vue.addEventListener('wheel',e=>{e.preventDefault();const r=vue.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=Math.exp(-e.deltaY*(e.ctrlKey?0.01:0.0015));
  const nz=Math.max(0.5,Math.min(400,z*f)),k=nz/z;tx=mx-(mx-tx)*k;ty=my-(my-ty)*k;z=nz;poser()},{passive:false});
vue.addEventListener('mousedown',e=>{gliss=[e.clientX-tx,e.clientY-ty];vue.style.cursor='grabbing';e.preventDefault()});
addEventListener('mouseup',()=>{gliss=null;vue.style.cursor='grab'});
addEventListener('mousemove',e=>{const r=vue.getBoundingClientRect(),p=D.pages[P];
  if(gliss){tx=e.clientX-gliss[0];ty=e.clientY-gliss[1];img.style.transform='translate('+tx+'px,'+ty+'px)'}
  const X=e.clientX-r.left,Y=e.clientY-r.top;cur=(X>=0&&Y>=0&&X<=r.width&&Y<=r.height)?[p.x0+(X-tx)/z,p.y0+(Y-ty)/z]:null;dessiner()});
addEventListener('resize',()=>ajuster());
$('prec').onclick=()=>page(P-1);$('suiv').onclick=()=>page(P+1);
const premiereEd=D.pages.findIndex(p=>p.ed),derniereCol=D.pages.map(p=>p.type!=='vierge').lastIndexOf(true);
$('b-debut').onclick=()=>page(0);$('b-ed').onclick=()=>page(premiereEd);$('b-fin').onclick=()=>page(derniereCol);
D.pages.filter(p=>D.lectures[p.num]&&!p.ed).map(p=>[p,sures(D.lectures[p.num]),D.lectures[p.num].mots.filter(m=>m.sur).length]).sort((a,b)=>b[1]-a[1]).slice(0,40)
  .forEach(([p,n,m])=>{const o=document.createElement('option');o.value=p.num-1;o.textContent='p. '+p.num+' · '+n+' lettres sûres'+(m?' · '+m+' mot'+(m>1?'s':''):'');$('mieux').appendChild(o)});
$('mieux').onchange=e=>{if(e.target.value!=='')page(+e.target.value);e.target.blur()};
$('gain').oninput=e=>{img.style.filter=e.target.value>1?'brightness('+e.target.value+')':''};$('gain').onchange=e=>e.target.blur();
let depart=premiereEd>=0?premiereEd:0;
try{const v=localStorage.getItem('livre_trad');if(v!==null)trad=+v%3}catch(e){}
try{const m=location.hash.match(/^#p(\d+)$/),s=localStorage.getItem('livre_page');if(m)depart=+m[1]-1;else if(s)depart=+s-1}catch(e){}
page(depart);
</script></body></html>
"""


if __name__ == "__main__":
    main()
