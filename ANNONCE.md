# Announcement and prize application (drafts)

Drafts: at the time of this commit neither text had been posted or submitted. The repository is
<https://github.com/wil-pe/paris4-title-search>. Both texts only state what FINDINGS.md supports.

## 1. Discord message (channel: show-and-tell)

Attach two or three figures: `figures/12_w126_before_after.jpg`,
`figures/05_ghost_emphai.jpg`, `figures/06_folds_vs_marks.jpg`.

> **PHerc. Paris. 4 title: a documented negative result, plus tools that run on a MacBook**
>
> Two days on one laptop (M4 Pro, 24 GB, no CUDA). No title found. Sharing everything so nobody
> has to redo it: https://github.com/wil-pe/paris4-title-search
>
> • `ink_9um` and the Paris 4 3D ink model running on Apple GPU (MPS) or CPU, reading only the
> needed chunks from the open bucket. `ink_9um` reproduces the published PHerc. 139 title map
> (r = 0.65) and reads the Paris 4 text; about 1 min per 2 × 2 cm tile.
> • The blank papyrus after the last column seen through five families of ink maps, including
> two we computed (3D model on the 1.129 µm scan over 35 cm², `ink_9um` over 282 cm²): only line
> ends and ghosts. Sensitivity is measured, not assumed: the Paris 4 3D model keeps 32–40 % of a
> real line once its ink is weakened to 40 % contrast, so a much fainter title would be missed.
> • Two ghosts read and matched to the edition: ΠΟΙ (col. 2, line 10) and ΕΜΦΑΙ (col. 1, line 3).
> • Three cheap tests for any title candidate, each a script: facing wrap, fold (mesh curvature),
> stability across checkpoints and depths (0.03 % of the blank zone survives, 11.8 % of real text).
> • Winding measured on the mesh: the text ends at the core, written side facing the core.
> • Static viewers to check a zone by eye, with millimetre rulers burnt into the images.
> • Segment w126-127: `ink_9um` gives clean letters where the published 2D map shows blobs
> (running text, already visible in the published 3D prediction; not a title).
> • Census of published segments: of the 20 First Letters scrolls, only PHerc. 800 has any.
>
> The same negative conclusion was reached in September in the thread "PHerc. Paris 4: complete
> recto/verso maps and a thirteen-depth atlas of the title region"; this is an independent
> confirmation with different tools. FINDINGS.md lists every number with its script and figure,
> and what was not verified. Scripts are commented in French; README and FINDINGS are in English.
> Feedback and corrections welcome.

## 2. Progress prize application

### What the prize page says (read on 13 November 2026)

Source: <https://scrollprize.org/prizes>, section "Progress Prizes", and the linked Google Form
"November 2026 Progress Prizes". In our own words:

- Submissions are judged monthly; the next deadline is 30 November 2026, 11:59 pm Pacific.
  Several submissions and several awards per month are allowed.
- One "best submission of the month" receives $20,000; the other awards are typically $20,000,
  $10,000, $5,000, $2,500, $1,000, $500 or $250, depending on significance.
- Preference goes to work that is released early, that actually gets used by the community, that
  improves results on real data, that fixes bugs in tools people use, and that is well
  documented.
- The page states: "We discourage standalone analytic, auditing or evaluation tools". Work that
  directly advances unwrapping or ink detection is what they want. Small contributions of one
  month should be bundled into one submission.
- Core requirements (collapsed section "Submission criteria and requirements"): address a
  specific problem on Vesuvius Challenge data, show a clear implementation and a demonstration,
  show an advantage over existing solutions, document it with usage examples, accept the
  community formats (OME-Zarr, tifxyz), keep output formats consistent, be modular.
- Terms: prizes are at the discretion of Scroll Prize, Inc.; the method must be open source under
  a permissive licence to accept a prize.
- The form asks for: e-mail, full name, team description, Discord display name (optional), the
  public URL of the contribution (a GitHub repository or pull request; required, several
  allowed), one free-text answer in four points (which scroll data, how it raises the probability
  of reading, what it enables that was not possible before, what evidence), and agreement to the
  terms.

Given the sentence quoted above, the application leads with what is a reusable tool (the three controls that apply to any candidate,
`ink_9um` on a millimetre window read from the bucket with safe loading of the weights; Apple GPU
support is already upstream and is not claimed,
the viewers for checking by eye, the before/after on real data), not with the negative result.

The title prize has its own rule: a title discovery must not be made public before the official
announcement. We found no title, so nothing in this repository is concerned.

### Text for the form's main question

"What is your contribution?" The form asks to mention four points; the answer follows them.

> **Contribution.** Open-source tools to run two official ink models on a laptop without CUDA,
> three false-positive controls that apply to any ink candidate, static viewers for checking a
> zone by eye, and a measured negative result on the expected title region of PHerc. Paris. 4.
> Repository: https://github.com/wil-pe/paris4-title-search (MIT). Inputs are the community formats (OME-Zarr volumes and surface
> volumes, tifxyz meshes) read from the open bucket. FINDINGS.md gives every number with the
> script and figure behind it and lists what was not verified.
>
> **(1) Scroll data.** PHerc. Paris. 4 (Scroll 1): segments `20260701183124-w010-027`,
> `20260623141924-w010-027` and `20260701183150-w126-127`, the 2.4 µm and 1.129 µm volumes, the
> published 3D and 2D ink predictions. PHerc. 139: the title segment, used as a known title.
> PHerc. 800: its six published patches. Public metadata of all samples for the segment census.
>
> **(2) How it raises the probability of reading.** (a) `ink9.py` runs `scrollprize/ink_9um` on
> Apple-silicon GPU or CPU, reading only the needed chunks from the bucket and loading the
> checkpoints in safe mode (`weights_only=True`): anyone with a recent Mac can test the
> multi-scroll model on any published surface volume in minutes, at about one minute per
> 2 × 2 cm tile. `balayage.py` does the same for `ink_paris4_3d_4.8um_gambiarra` on both Scroll 1
> scans. (b) Three scripted controls (ink of the facing wrap, folds of the flattened scroll from
> mesh curvature, stability across checkpoints and depth offsets) remove the false positives
> that cost us most of our time, and that the title prize asks submitters to rule out. They are
> set up for the end segment of Scroll 1; the facing-wrap and fold computations take any tifxyz
> mesh, and the stability test any set of `ink9.py` outputs. (c) Static viewers
> (`visionneuse_pages.py`, `visionneuse_zone.py`, `visionneuse_livre.py`) show the same place
> through several ink maps, with millimetre rulers burnt into the images, so that a human can
> check a zone and report a place with a screenshot; the letters of section 8 of FINDINGS.md
> were found that way. (d) The
> census (`catalogue/segments.csv`) shows where ink models can be tried today: among the 20
> First Letters scrolls only PHerc. 800 has published segments.
>
> **(3) What was not possible before.** To our knowledge: running the Paris 4 3D model without
> a CUDA machine (the official inference code calls `.cuda()`); running `ink_9um` on a window
> given in millimetres and read straight from the bucket, with repeated depth-offset passes
> (Apple GPU support itself already exists upstream: PR #1865 in ScrollPrize/villa); an `ink_9um` map of the whole
> blank end of Scroll 1 (282 cm²) and a 3D-model map of part of it on the 1.129 µm scan
> (35 cm²); a measured sensitivity for a
> title search (the 3D model keeps 32–40 % of a real line once its ink is weakened to 40 %
> contrast in the raw scan, and does not transfer to the PHerc. 139 title); the fold test and the
> stability test as reusable scripts. The negative result itself confirms, with different tools,
> the one posted in show-and-tell on 11 September 2026.
>
> **(4) Evidence.** On real data only: our `ink_9um` port reproduces the published PHerc. 139
> title map (r = 0.65) and reads the known text of Scroll 1 (two checkpoints agree at r = 0.96);
> both are sanity checks of the port on training scrolls of that model, not held-out validation;
> on segment w126-127 it gives legible letters where the published 2D map shows blobs (figure
> 12; the published 3D prediction also shows them); two ghosts are read and matched to the
> edition (figures 4 and 5); the stability test leaves 0.03 % of the blank zone against 11.8 %
> of real text (figure 9); outside ghost zones, detectors on four map families find about as
> many windows as chance predicts (21 for 16 expected, 9 for 7, 17 for 15), except for border
> artefacts of one map, and none of them looks like writing. Limits are stated in FINDINGS.md: no
> papyrologist has reviewed the readings, the repository was not rerun from a clean clone, and
> script comments are in French.

### Other fields of the form

- **Team description**: individual submission, with AI assistance (Claude Code) for the code and
  the write-up. Adjust if someone else took part.
- **URL**: <https://github.com/wil-pe/paris4-title-search>.
- E-mail, full name and Discord display name are yours to fill in.
