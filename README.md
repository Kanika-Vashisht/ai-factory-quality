# factory-quality-ai

Visual defect detection on industrial parts and remaining-useful-life (RUL) prediction for
machines, both tested for how well they hold up when conditions change.

> Status: **Phase 0** – repo skeleton, dataset download scripts and data loaders.
> Models, serving and deployment come in later phases (see [Roadmap](#roadmap)).

## Why this matters

**Defect detection.** Most factories still rely on manual visual inspection or rule-based
machine vision at the end of the line. People get tired and miss things; hand-written rules
break as soon as a part or the lighting changes. A defect that slips through becomes scrap,
a warranty claim or a recall, all of which cost far more than catching it at the station.
The hard part is that defects are *rare* and *varied*: you can collect thousands of good parts,
but only a handful of each kind of scratch or crack, and new kinds show up that nobody has
seen before. That is why this project treats inspection as **anomaly detection** – learn what
"normal" looks like from good parts only, and flag anything that deviates.

**Predictive maintenance.** Unplanned downtime – a machine stopping without warning – halts
the line, idles the workers and often damages the part being made. Fixed-interval maintenance
avoids some of this but replaces components that still had plenty of life left. Predicting the
**remaining useful life** from sensor data lets maintenance be scheduled just before it is
needed. Here the dangerous mistake is predicting *too much* remaining life, because that is the
case where the machine fails before anyone acts.

**Distribution shift.** Both models are trained in one setting and used in another: a camera
gets refocused, the lighting changes, a machine runs at a different operating point. A model
that only works on its own test split is not much use on a real shop floor, so robustness to
these changes is measured explicitly in this project.

## Data

| Dataset | Used for | Size | License |
|---|---|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Defect detection | ~4.9 GB total, 100–700 MB per category | CC BY-NC-SA 4.0 |
| [NASA C-MAPSS](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) | RUL prediction | ~12 MB | NASA open data |

**MVTec AD** has 15 categories of objects and textures. The training set contains only
defect-free images; the test set has good and defective images, with a pixel mask for every
defect. This project uses five categories by default (bottle, metal_nut, screw, transistor,
cable – set in `configs/data.yaml`).

**C-MAPSS** contains simulated run-to-failure data of turbofan engines: 21 sensor channels and
3 operational settings per cycle. The four subsets differ in operating conditions and fault
modes, which gives a natural train/test shift (train on FD001, test on FD002/FD004):

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 | 2 (HPC + fan degradation) |
| FD004 | 249 | 248 | 6 | 2 (HPC + fan degradation) |

Raw data is never committed. Download it with the scripts below.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"

python scripts/download_cmapss.py                  # ~12 MB
python scripts/download_mvtec.py                   # 5 default categories, ~1.4 GB
python scripts/download_mvtec.py --categories screw # or just one category

pytest
ruff check .
```

Data is stored in `data/raw/` (git-ignored):

```
data/raw/
├── mvtec/<category>/{train,test,ground_truth}/...
└── cmapss/{train,test,RUL}_FD00x.txt
```

Loading it in Python:

```python
from fqa.config import data_dir
from fqa.vision.mvtec import list_samples, load_image, load_mask
from fqa.maintenance.cmapss import load_subset

samples = list_samples(data_dir() / "mvtec", "screw", "test")
image, mask = load_image(samples[0]), load_mask(samples[0])

train, test = load_subset(data_dir() / "cmapss", "FD001")  # both have an `rul` column
```

## Project structure

```
src/fqa/
├── config.py        # reads configs/*.yaml
├── download.py      # download, checksum and safe extraction helpers
├── vision/          # MVTec AD loading (models from Phase 1)
└── maintenance/     # C-MAPSS loading (models from Phase 4)
configs/             # dataset paths and category selection
scripts/             # dataset download scripts
tests/               # unit tests on tiny generated fixtures (no downloads needed)
```

## Roadmap

- [x] Phase 0 – Skeleton and data
- [ ] Phase 1 – Vision baselines (autoencoder, supervised few-shot classifier)
- [ ] Phase 2 – PatchCore and PaDiM
- [ ] Phase 3 – Vision robustness experiment
- [ ] Phase 4 – Predictive maintenance (RUL) with shift experiment and uncertainty
- [ ] Phase 5 – Experiment tracking and explainability
- [ ] Phase 6 – Serving (ONNX, FastAPI, Postgres)
- [ ] Phase 7 – CI and drift monitoring
- [ ] Phase 8 – Orchestrated retraining
- [ ] Phase 9 – Kubernetes deployment (local)
- [ ] Phase 10 – Polish

## Citations

MVTec AD is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by MVTec Software GmbH.
It may not be used commercially. If you use it, please cite:

- Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger.
  *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised
  Anomaly Detection.* International Journal of Computer Vision 129(4), 1038–1059, 2021.
- Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
  *MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

C-MAPSS comes from the NASA Prognostics Center of Excellence data repository:

- A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set.*
  NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- A. Saxena, K. Goebel, D. Simon, N. Eklund. *Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation.* International Conference on Prognostics and Health Management
  (PHM), 2008.
