# Looking for the title of PHerc. Paris. 4: tools, controls and a documented negative result

Two days of work (12–13 November 2026) on the [PHerc. Paris. 4 title prize](https://scrollprize.org/prizes)
of the Vesuvius Challenge, done on a single laptop. We did **not** find the title. What this
repository offers instead:

- **Laptop inference for two official ink models.** `scripts/ink9.py` runs the multi-scroll
  model `scrollprize/ink_9um`, and `scripts/balayage.py` runs the Paris 4 3D model
  `scrollprize/ink_paris4_3d_4.8um_gambiarra`, on an Apple-silicon GPU (PyTorch MPS) or on CPU.
  Both read only the chunks they need from the public bucket. For the 3D model the official
  inference code calls `.cuda()`; for `ink_9um` the official script already falls back to CPU,
  and `ink9.py` adds the Apple GPU and a window given in millimetres.
- **Three controls every title candidate should pass** before anyone gets excited: the
  facing-wrap test (ghosts), the fold test, and the stability test. Each one is a script.
- **A measured negative result** on the blank papyrus after the last column, with the
  sensitivity of each detector stated, and two "ghosts" read and matched to the published edition.
- **A census** of every published segment in the open-data bucket, with what is needed to run
  an ink model on it.

Everything that is claimed is in [FINDINGS.md](FINDINGS.md), with the script and the figure
behind each number, and a list of what was not verified.

Prior work: the same conclusion (no legible title, and "letters" in the blank zone that are echoes
of line ends one wrap away) was reported two months earlier in the Discord thread *"PHerc. Paris 4:
complete recto/verso maps and a thirteen-depth atlas of the title region"* (11 September 2026),
repository <https://gitlab.com/yohanntschudi/phercparis4-depth-echoes>. Our work is independent of
that code and arrives at the same place by a different route.

Comments, docstrings and console output of the scripts are in French; this README and
FINDINGS.md give the English entry points.

## Hardware

MacBook Pro, Apple M4 Pro, 24 GB of unified memory, macOS 27. No CUDA GPU, no cloud compute.
Measured on that machine:

| Task | Cost |
|---|---|
| `ink_9um`, one 2048 × 2048 px tile at 9.6 µm (19.7 × 19.7 mm), two checkpoints | about 60 s, of which about 20 s of download (run log) |
| `ink_9um` on the whole blank zone, 127 × 167 mm (63 tiles) | 64 min |
| Paris 4 3D model, one 128³ cube | 0.085 s on MPS, 0.26 s on CPU (session notes) |
| Paris 4 3D model, sweep of 60 × 58 mm on the 1.129 µm scan (380 tiles of 3.1 mm) | 4 h 20 min overnight, including a 1 h 40 min stall, about 150 GB of network traffic |
| Pale-writing detector on published maps (`detecteur_carte.py`) | 16 min on CPU |

The download speed of the bucket (about 10 MB/s here), not the GPU, was the bottleneck.

## Install

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

The two official models live in the ScrollPrize repositories; the scripts expect their code under
`externe/` and their weights under `modeles/` (both git-ignored):

```bash
git clone --filter=blob:none --branch ink-paris4-3d-gambiarra https://github.com/ScrollPrize/villa externe/villa
```

```bash
git clone --filter=blob:none --branch merge-ink-pipelines https://github.com/ScrollPrize/villa externe/villa-ink9
```

We used commit `c9ca2dc` of the first branch and `47a9c37` of the second.

The weights come from Hugging Face (both models are MIT). The Paris 4 3D model (98 MB):

```bash
mkdir -p modeles/ink_paris4_3d_4.8um_gambiarra && curl -L -o modeles/ink_paris4_3d_4.8um_gambiarra/model.safetensors https://huggingface.co/scrollprize/ink_paris4_3d_4.8um_gambiarra/resolve/main/model.safetensors
```

and the two final checkpoints of `ink_9um` (138 MB each; note the local file name, which joins
the folder and file names of the Hugging Face repository with a dash):

```bash
mkdir -p modeles/ink_9um && for s in 42 43; do curl -L -o modeles/ink_9um/hybrid_3d2d-seed$s-step-075000.pth https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed$s/step-075000.pth; done
```

Only for the two text tools, the Digital Corpus of Literary Papyri:

```bash
git clone --filter=blob:none https://github.com/papyri/idp.data externe/idp.data
```

Then fetch the small input files (two meshes, about 360 MB, and the scan-to-scan transforms):

```bash
.venv/bin/python -I scripts/preparer.py --executer
```

Run without `--executer`, the same command only lists what is missing. All scripts are run from
the repository root as `.venv/bin/python -I scripts/<name>.py`. Downloads are cached in `data/`;
set `LABO_DATA=/some/other/folder` to keep the cache elsewhere.

## Conventions

- "Segment 701" is `PHercParis4/segments/20260701183124-w010-027`, the innermost 18 wraps
  (408 × 167 mm once flattened). Positions are given in millimetres in its flattened frame:
  x along the sheet in reading order, y downwards; 1 px of the 2.4 µm surface volume = 0.0024 mm.
- The last column of text spans x ≈ 217–280 mm; x > 281 mm is the blank papyrus where a final
  title is expected.
- Along the mesh normal, t > 0 points away from the scroll axis (checked by
  `scripts/sens_enroulement.py`).

## Tools

### Ink models on a laptop

```bash
.venv/bin/python -I scripts/ink9.py titre139
```

runs `ink_9um` on the PHerc. 139 title segment (50 × 17 mm) and writes one probability map per
checkpoint in `data/ink9/`. `scripts/accord_titre139.py` then compares them with the published
ink map (we get r = 0.65).

```bash
.venv/bin/python -I scripts/ink9.py paris4 p4_texte 5 45 215 250 -8,0,8
```

runs it on a window of segment 701 given in mm (`Y0 Y1 X0 X1`), here the last column of text, at
three depth offsets (in 2.4 µm layers). Large windows are processed tile by tile and can be
interrupted and resumed. Two other modes: `segment NAME SURFACE_VOLUME_PATH Y0 Y1 X0 X1` for any
other 2.4 µm surface volume of the bucket, and `natif NAME SURFACE_VOLUME_PATH` for surface
volumes that are natively at about 9 µm (PHerc. 800, see `scripts/ink9_pherc800.sh`).
`INK9_POIDS="*"` uses every checkpoint present instead of the two default ones. Checkpoints are
loaded with `torch.load(..., weights_only=True)`.

```bash
.venv/bin/python -I scripts/balayage.py scan fin_1129 20000 44320 118750 144350 11 0/1
```

sweeps a window of segment 701 (bounds in 2.4 µm pixels) with the Paris 4 3D model on the
1.129 µm scan (`11`) or the 2.4 µm scan (`24`), reading only the chunks near the sheet.
`scripts/nuit.sh 4` runs the same sweep with four processes that restart every ten tiles;
`scripts/mosaique_balayage.py` assembles the tiles. `balayage.py valider` measures what the fast
path costs in fidelity, and `balayage.py etalon ...` reruns the model after weakening the ink in
the raw scan (see FINDINGS.md, sensitivity). `scripts/modele_encre.py` alone prints the time per
cube on MPS and CPU.

Practical lesson for MPS: keep the batch shape fixed. With variable batch sizes one process grew
to 17 GB and the machine swapped for 1 h 40 min; with fixed shapes, `torch.mps.empty_cache()`
after each tile and a fresh process every ten tiles, each process stays at 4–5 GB.

### The three controls

```bash
.venv/bin/python -I scripts/voisinage.py data/PHercParis4/segments/20260701183124-w010-027/mesh/20260701183124-on-20260411134726-2.4um.tifxyz data/PHercParis4/carte/voisinage.npz
```

**1. Facing wrap.** For every point of the segment, finds the point of the adjacent wrap that
faces it (0.1 mm away). `carte_fin_paris4.py` projects the published 3D ink predictions into
depth slices, `carte_sans_fantomes.py` removes what the facing wrap explains, and
`planches_candidats.py` draws what is left.

```bash
.venv/bin/python -I scripts/pliures.py carte
```

```bash
.venv/bin/python -I scripts/pliures.py controle 337 42
```

**2. Fold.** Measures the curvature of the sheet on the mesh and tells whether a candidate at
(x, y) mm sits on a fold of the flattened scroll. `pliures.py superposer` draws the folds over
any 9.6 µm map.

```bash
.venv/bin/python -I scripts/stabilite.py p4_droite --extrait p4_vierge 30 64
```

**3. Stability.** Takes every pass of `ink9.py` over a zone (checkpoints × depth offsets) and
keeps the pixel-wise minimum: what is ink in all of them.

### Pale-writing detectors

```bash
.venv/bin/python -I scripts/detecteur_carte.py ink9
```

A small network learns to tell blank papyrus from the same papyrus with real letters of the
scribe superimposed and dimmed. Each 8 mm band is scored by a network that never saw it as a
"blank" example, so a faint title cannot be learned as background. Without argument it works on
the maps of the two published 3D models; `canon` and `fin2d` use the two published 2D maps
(`preparer.py --cartes` downloads them), `ink9` uses the outputs of `ink9.py`.
`detecteur_pale.py` (1.5 mm windows) and `detecteur_mosaique.py` (4.9 mm windows) do the same on
the tiles of the 1.129 µm sweep; they need `scripts/assembler_etalon.py` first.

### Geometry and physics

```bash
.venv/bin/python -I scripts/sens_enroulement.py
```

prints the distance to the scroll axis along the unrolled sheet and the orientation of the mesh
normal. `scripts/relief.py NAME Y0 Y1 X0 X1` (mm) measures the density profile across the sheet,
with and without ink, on the raw scan. `double_energie.py` and `contraste_energies.py` compare
the 78 keV and 137 keV scans on a block; `etalonnage_139.py` and `etalonnage_paris4.py` are the
earlier, slower versions of the ink-weakening calibration; `pilote_1129.py` is the pilot that
compared our inference with the published prediction.

### Census and data access

```bash
.venv/bin/python -I scripts/recenser_segments.py
```

writes `catalogue/segments.csv` (included: 710 rows, 324 segments, 14 samples) from the public
`metadata.min.json`, reading only Zarr headers. `construire_catalogue.py` does the same for
volumes, `lister_bucket.py PHerc0139/segments/` lists a bucket folder, and
`telecharger_volume.py PHerc0125 --niveau 5` shows (and with `--executer` downloads) one
resolution level of a volume. `catalogue/statut_lecture.csv` is our hand-kept table of which
scroll is eligible for which prize, copied from the prize page on 12 November 2026: check it
against the current page before relying on it.

### Text tools

```bash
.venv/bin/python -I scripts/recherche_texte.py 989336
```

looks for a passage shared between the edited text of PHerc. Paris. 4 (DCLP 989336) and any other
literary papyrus of the DCLP. `scripts/voisins_bibliotheque.py` ranks the Herculaneum papyri of
the DCLP by vocabulary similarity with it.

### Viewers

```bash
.venv/bin/python -I scripts/visionneuse_pages.py
```

builds a static page-by-page viewer of the blank zone in `apercus/visionneuse/` (34 pages of
40 × 30 mm, five techniques, millimetre rulers burnt into the images so that a screenshot can be
located). `visionneuse_zone.py` builds a zoomable map of the whole blank zone,
`comparaison_titre.py` the seven-panel comparison of figure 1, and `visionneuse_livre.py` a
"book" view of the whole unrolled scroll, one column per page, next to the edition. Serve them
with `python3 -m http.server 8765 --directory apercus`. They read the outputs of the steps above
and published maps that must already be on disk; their docstrings list the inputs.

### Shared modules and launchers

`volume.py` reads a box of any OME-Zarr volume of the bucket by downloading only the chunks it
needs (with a local cache), and reads rectangles of large tiled TIFFs. `rendu_surface.py`
projects a volume (scan or 3D ink prediction) along the normals of a tifxyz mesh window, with
optional depth slices; it can be used alone (see its docstring). `ink9_paris4.sh`,
`ink9_pherc800.sh` and `nuit.sh` are the launchers of the runs reported in FINDINGS.md.

`recherche_mot.py` and `recherche_faible.py` are two earlier searches (template matching of a
known word, and averaging of sub-threshold scores), kept because FINDINGS.md quotes their numbers.

## What is not in the repository

No scan data, no model weights and no official code are redistributed. The twelve small figures
in `figures/` are derived from Vesuvius Challenge data and carry the attribution below.

## Licence and attribution

- **Code** (`scripts/`): MIT licence, see `LICENSE`.
- **Data**: the scans, segments and published ink predictions belong to the Vesuvius Challenge
  and are used under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The figures
  and `catalogue/*.csv` are derived from them and are shared under the same terms. As asked on
  <https://scrollprize.org/data>, we cite the dataset we used: Giorgio Angelotti, Stephen
  Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro
  Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader,
  C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri.*
  Vesuvius Challenge.
- **Models and model code**: `scrollprize/ink_paris4_3d_4.8um_gambiarra` (MIT) and
  `scrollprize/ink_9um` (MIT), run with the code of <https://github.com/ScrollPrize/villa> (MIT).
- **Greek text**: record 989336 of the Digital Corpus of Literary Papyri
  (<https://github.com/papyri/idp.data>, © DCLP and LDAB, CC BY 3.0). The record refers to
  F. Nicolardi, S. Parsons, D. Delattre, G. Del Mastro, R. L. Fowler, R. Janko, T. Reinhardt,
  C. S. Parker, C. Chapman and W. B. Seales, *ZPE* 229 (2024), pp. 1–13, and to F. Nicolardi,
  D. Delattre, G. Del Mastro, R. Fowler and R. Janko, *Cronache Ercolanesi* 54 (2024),
  pp. 9–27; full references at the end of FINDINGS.md.
