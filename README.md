# FedSP-PEFT — Privacy-Preserving Federated Story Point Estimation

A federated learning system that predicts agile **story points** from JIRA issue
text, where **each client is one software project**, so raw issue data never
leaves the project. Built for a bachelor thesis on applying federated learning
to story point estimation (SPE) — a novel intersection — with
parameter-efficient fine-tuning.

**Task:** 5-class **ordinal** classification over the Fibonacci story-point deck
`{1, 2, 3, 5, 8}` (not regression). The ordinal structure is exploited by a CORN
loss head; MAE on class values and quadratic-weighted Cohen's κ are the primary
metrics (bridging to the regression-based SPE literature).

> Accuracy on SPE is inherently low for everyone — even sophisticated deep models
> barely beat naive baselines (Tawosi et al. 2023). The contribution is the
> **local-only → federated → centralized gap analysis** under a privacy
> constraint, the **personalization** finding, and **communication cost** — not
> absolute accuracy.

---

## Contents

1. [Install](#install)
2. [Dataset — TAWOS, and how to export it](#dataset--tawos-and-how-to-export-it)
3. [Training](#training)
4. [Every training flag](#every-training-flag)
5. [Inference](#inference)
6. [Framework overview](#framework-overview-fedsp-peft)
7. [Scripts & project structure](#scripts)

---

## Install

```powershell
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is strongly recommended for `codebert-base` runs; the
`bert-tiny` smoke run works on CPU (`--device cpu`).

---

## Dataset — TAWOS, and how to export it

### The source dataset

This project uses **TAWOS** — Tawosi, Al-Subaihin, Moussa & Sarro, *"A Versatile
Dataset of Agile Open Source Software Projects"*, MSR 2022. It is a MySQL dump of
JIRA issues from real open-source projects, with story points as assigned by the
actual development teams.

- Paper / repo: <https://github.com/SOLAR-group/TAWOS>
- You need the **MySQL dump restored to a local database**. This repo does not
  ship the data — `data_to_train_on/` is gitignored.

The relevant schema is two tables, `Issue` and `Project`, joined on
`Issue.Project_ID = Project.ID`. Only these `Issue` columns are read:
`Issue_Key, Title, Description, Story_Point, Type, Priority, Creation_Date`.

### Step 1 — point the exporter at your database

`export_issues.py` reads credentials from a `.env` file in the project root (it
is gitignored — never commit it):

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=tawos
```

Missing variables fail fast with an explicit message. Extra dependencies for
this step only: `mysql-connector-python` and `python-dotenv`. `mysql.connector`
is imported lazily, so training never needs a database driver.

### Step 2 — run the export

```powershell
python export_issues.py
```

There are **no CLI flags** — selection, cleaning and output rules are fixed so
the export stays reproducible and citable in the thesis.

**Project selection** (the canonical filtering query):

```sql
SELECT p.Name, COUNT(*) AS Total_Issues, COUNT(DISTINCT i.Story_Point) AS Num_Classes
FROM Issue i JOIN Project p ON i.Project_ID = p.ID
WHERE i.Story_Point IN (1, 2, 3, 5, 8)
GROUP BY p.Name
HAVING COUNT(*) >= 500 AND COUNT(DISTINCT i.Story_Point) >= 4
ORDER BY Total_Issues DESC;
```

A project qualifies only if it has **at least 500 issues** on the Fibonacci deck
and **at least 4 of the 5 classes** present. Issues are then pulled per project
ordered by `Creation_Date ASC`, which is what makes the temporal split possible
later.

**Cleaning happens here, once, at export time.** The training pipeline consumes
these CSVs as-is and only *validates* them. In order:

1. Strip one layer of wrapping double quotes (a legacy export artifact)
2. HTML: unescape entities, then strip tags
3. Jira markup: `{code}` / `{noformat}` blocks become `[CODE]`; known macros
   (`{color}`, `{panel}`, …) stripped by name; wiki headings `h1.`–`h6.`
   stripped; bullet markers removed; `*bold*` unwrapped; `||` table markers removed
4. URLs become `[URL]`; issue keys like `ABC-123` become `[ISSUE_REF]`
5. Whitespace normalisation
6. Length floor: combined cleaned title+description under 10 chars drops the row
7. `Priority` normalised to `{Highest, High, Medium, Low, Lowest, Unknown}`;
   `Type` stripped, NaN to `Unknown`; `Story_Point` cast to int

Deliberately **not** done (documented thesis decisions): no lowercasing, no
stopword removal, no stemming, no number stripping — the encoder is cased and
`NullPointerException` carries signal. Rows with a missing `Description` are
**kept** as title-only.

**Output:**

```
data_to_train_on/
  Apache_Mesos.csv            one CSV per qualifying project
  Appcelerator_Studio.csv     columns: Issue_Key, Title, Description,
  ...                                  Story_Point, Type, Priority, Creation_Date
  preprocessing_report.json   per-project + global counts (rows in/out, drops,
                              substitutions) — cited verbatim in the thesis
```

Reference figures from the June 2026 export: **19 projects, 42,002 rows in,
41,995 out** (7 dropped by the length floor), with 8,862 `[CODE]`, 11,480 `[URL]`
and 6,513 `[ISSUE_REF]` substitutions. A large deviation from these numbers
suggests a regex regression.

### Step 3 — that's it

`fl/data.py::validate_cleaned_dataframe` checks at load time that it was handed
*cleaned* CSVs (no `{code` remnants, no quote-wrapped titles, canonical
Priority/Story_Point values) and fails loudly pointing back at `export_issues.py`.
If you already have the cleaned CSVs, point `--data-dir` at them and skip this
section entirely.

---

## Training

`train_federated_dl.py` runs the whole pipeline in one process:

```
warm-start (pre-train on the largest project, then exclude it from the pool)
  -> classic baselines (TF-IDF+SVM, median-SP, per project)
  -> local-only (each client alone)
  -> centralized (pooled, privacy ignored — upper bound)
  -> federated (FedProx/FedAvg rounds)
  -> per-project evaluation, summary tables, communication cost
```

Every condition trains its full budget and returns its **best-on-validation**
checkpoint — no early stopping anywhere — so the conditions stay comparable.

### Smoke test (plumbing only — never interpret its accuracy)

```powershell
python train_federated_dl.py --data-dir data_to_train_on `
  --model-name prajjwal1/bert-tiny --max-length 64 `
  --rounds 2 --local-epochs 1 --warmstart-epochs 1 `
  --skip-centralized --skip-local-only `
  --save-dir artifacts_smoke --seed 42
```

Runs in minutes. `bert-tiny` collapses to predicting roughly one class at this
budget, so distinct conditions can produce identical metrics — that is expected.
The only thing this run proves is that the plumbing works.

### A full single run (the method defaults used for results)

```powershell
python train_federated_dl.py --data-dir data_to_train_on `
  --model-name microsoft/codebert-base --max-length 256 `
  --rounds 60 --local-epochs 1 --batch-size 64 `
  --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
  --prox-mu 1e-2 --split-mode temporal --head-type corn `
  --checkpoint-every 5 --save-dir artifacts_shared --seed 42
```

Swap in `--personalized-head --generic-head` for the FedSP-PEFT-P variant, or
`--no-lora` for the full-fine-tuning comparison.

On bash, replace the trailing backticks with `\` for line continuation.

Interrupted? Re-run the **same command** plus `--resume`. Finished phases are
skipped and the warm-start is loaded from cache; resume is bit-reproducible with
respect to client selection and data sampling.

### Multi-seed runs + statistics

`run_experiments.py` wraps the above across seeds, running the FedProx condition
(with baselines) and the FedAvg condition (`--prox-mu 0`, reusing that seed's
baselines) per seed. Unrecognised flags are forwarded verbatim to both runs.

```powershell
python run_experiments.py `
  --seeds 42-44 --results-root experiments_temporal_shared `
  --data-dir data_to_train_on `
  --model-name microsoft/codebert-base --max-length 256 `
  --rounds 60 --local-epochs 1 --batch-size 64 `
  --lr 3e-5 --warmstart-lr 3e-5 --warmstart-epochs 10 `
  --prox-mu 1e-2 --split-mode temporal --head-type corn `
  --checkpoint-every 5 --resume

python compute_statistics.py --experiments-root experiments_temporal_shared --metric mae
python compute_statistics.py --experiments-root experiments_temporal_shared --metric cohen_kappa
python compute_statistics.py --experiments-root experiments_temporal_shared --metric macro_f1
```

Output lands in `<results-root>/seed_<N>/{fedprox,fedavg}/results/*.json` plus a
`manifest.json`. Statistics (Wilcoxon signed-rank, Friedman, Nemenyi post-hoc,
Vargha-Delaney Â / Cliff's δ) are written to `<results-root>/statistics/`.
Pairing is **per (seed, project)** — pooled "global" entries are ignored on load.

Runner-only flags: `--seeds`, `--prox-mu`, `--results-root`, `--skip-fedavg`,
`--skip-baselines`, `--force`, `--dry-run`. Everything else is passed through to
`train_federated_dl.py`.

### Leave-one-project-out onboarding

Train with a project held out, then measure how much of its own history a new
project needs before joining pays off:

```powershell
python train_federated_dl.py --data-dir data_to_train_on `
  --holdout-project Hyperledger_Sawtooth --personalized-head --generic-head `
  --head-type corn --split-mode temporal --save-dir artifacts_lopo

python run_lopo.py --artifact-dir artifacts_lopo/federated `
  --data-dir data_to_train_on --holdout-project Hyperledger_Sawtooth `
  --head-init generic --budgets 0,10,25,50,100 --out-dir results

python compare_lopo.py --results-dir results --out-csv lopo_crossover.csv
```

**See [`commands.txt`](commands.txt) for the full command runbook.**

---

## Every training flag

All flags of `train_federated_dl.py`, with defaults in parentheses.

### Required / core

| Flag | Default | What it does |
|---|---|---|
| `--data-dir` | *(required)* | Folder holding one CSV/XLSX per project. Must be `export_issues.py` output — raw data is rejected at load time. |
| `--model-name` | `prajjwal1/bert-tiny` | HuggingFace encoder id. Results use `microsoft/codebert-base`; `bert-tiny` is a plumbing model only. |
| `--save-dir` | `artifacts` | Where trained artifacts and checkpoints are written. |
| `--seed` | `42` | Master seed for splits, init, client selection and sampling. |
| `--device` | `cuda` | `cuda` or `cpu`. |

### Data & splitting

| Flag | Default | What it does |
|---|---|---|
| `--max-length` | `128` | Token cap for `title [SEP] description`. 128 truncates the longest 10–15% of issues; results runs use 256. |
| `--batch-size` | `16` | Mini-batch size for every training phase. |
| `--test-size` | `0.2` | Per-client fraction held out as test. Evaluated **once** per condition. |
| `--val-size` | `0.1` | Per-client fraction carved from train as validation — drives best-on-val model selection in *every* condition. |
| `--split-mode` | `random` | `random` = stratified by story point. `temporal` = train on each client's earliest issues, validate/test on its latest (the realistic setting; primary for results). |

### Optimization

| Flag | Default | What it does |
|---|---|---|
| `--lr` | `2e-5` | Learning rate for local and centralized training. |
| `--weight-decay` | `1e-4` | AdamW weight decay. |
| `--rounds` | `8` | Federated communication rounds. Also sets the local-only budget (`rounds × local-epochs`) so per-client exposure matches. |
| `--local-epochs` | `1` | Epochs each selected client trains per round. |

### Federated behaviour

| Flag | Default | What it does |
|---|---|---|
| `--prox-mu` | `1e-2` | FedProx proximal strength `(mu/2)·‖w − w_global‖²`, applied over aggregatable params only. **`0` degrades to FedAvg** and the run is auto-labeled as such. |
| `--clients-per-round-fraction` (alias `--fraction`) | `1.0` | Fraction of clients sampled per round. `1.0` = full participation. |
| `--local-sample-ratio-per-epoch` | `1.0` | Fraction of each client's train set used per local epoch (sub-sampling for speed). |
| `--sample-with-replacement` | off | Draw that per-epoch sample with replacement. |

### Head & personalization

| Flag | Default | What it does |
|---|---|---|
| `--head-type` | `ce` | `ce` = 5-logit softmax + class-weighted CrossEntropy. `corn` = 4-logit CORN ordinal head + `corn_loss`, which penalizes distant story-point misses more than adjacent ones. Results runs use `corn`. |
| `--personalized-head` | off | **FedSP-PEFT-P**: the head stays local to each client and is never aggregated; only LoRA-B and the embeddings are federated. No pooled "global" metric is emitted in this mode — per-project only. |
| `--generic-head` | off | Personalized mode only: additionally save a one-way weighted average of client heads as `generic_head.pt`, used *only* to initialize new clients. Never pushed back to participants. |

### LoRA / parameter efficiency

| Flag | Default | What it does |
|---|---|---|
| `--no-lora` | LoRA on | Disable LoRA and fully fine-tune + federate the whole encoder (the comparison arm for parameter efficiency). |
| `--no-ffa-lora` | FFA on | Train and aggregate LoRA-A as well. The default keeps **A frozen**, so `avg(B·A) = avg(B)·A` and aggregation is exact. |
| `--freeze-encoder` | off | Freeze the encoder entirely — gradients stop at the head. The cheapest run available. |
| `--lora-r` | `8` | LoRA rank. |
| `--lora-alpha` | `16` | LoRA scaling factor. |
| `--lora-dropout` | `0.05` | Dropout inside the LoRA adapters. |
| `--lora-target-modules` | `query value` | Modules to adapt. Correct for BERT/RoBERTa (including `bert-tiny` and `codebert-base`). **DistilBERT needs `--lora-target-modules q_lin v_lin`.** |

### Warm-start

| Flag | Default | What it does |
|---|---|---|
| `--warmstart-project` | `lsstcorp` | Project centrally pre-trained on, giving every condition the same starting checkpoint. It is then **excluded from the FL client pool**. |
| `--warmstart-epochs` | `10` | Warm-start epoch budget. |
| `--warmstart-patience` | `3` | Early-stopping patience on the warm-start validation split. |
| `--warmstart-lr` | `2e-5` | Warm-start learning rate. |
| `--warmstart-val-size` | `0.15` | Validation fraction inside the warm-start project. |
| `--run-no-warmstart-fl` | off | Run a **second** federated pass from random init with identical client selection, isolating the warm-start's effect. Roughly doubles federated time. |

### Choosing which conditions to run

| Flag | Default | What it does |
|---|---|---|
| `--skip-centralized` | off | Skip the pooled upper-bound condition. |
| `--skip-local-only` | off | Skip the per-client no-federation condition. |
| `--skip-classic-baselines` | off | Skip the per-project TF-IDF+LinearSVM and median-SP baselines. They run before any deep training (about 2 min, CPU) and are RNG-neutral. |
| `--holdout-project` | `None` | Exclude a project from the pool **entirely**, reserving it as an unseen external client for the onboarding experiment. Cannot be the warm-start project. |

### Checkpointing & logging

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint-every` | `0` (off) | Checkpoint frequency. The unit is **epochs** for centralized/local-only and **global rounds** for federated. |
| `--checkpoint-keep` | `2` | How many numbered checkpoints to retain; `latest/` and `best/` are always kept. |
| `--resume` | off | Auto-detect and resume the newest checkpoint under `<save-dir>/checkpoints/`. Refuses to resume across an incompatible config (different head type, encoder, or personalization mode). |
| `--resume-from` | `None` | Resume from an explicit checkpoint directory, overriding auto-detection. |
| `--central-log-every` | `1` | Log the centralized condition every N epochs. |

---

## Inference

Shared-head runs save `<save-dir>/federated/{model_state.pt, metadata.json, tokenizer/}`;
personalized runs save `<save-dir>/federated/{shared_state.pt, heads/<project>.pt, generic_head.pt?}`.

```powershell
# shared-head artifact
python predict_saved_model.py --artifact-dir artifacts_shared/federated `
  --data-dir data_to_test_on --out-csv predictions.csv

# personalized artifact — overlay one project's own head
python predict_saved_model.py --artifact-dir artifacts_fedsp_peft_p/federated `
  --data-dir data_to_test_on --head-project Apache_Mesos `
  --out-csv predictions_mesos.csv

# personalized artifact — a brand-new project with no head of its own
python predict_saved_model.py --artifact-dir artifacts_fedsp_peft_p/federated `
  --data-dir data_to_test_on --generic-head --out-csv predictions_newproject.csv
```

| Flag | Default | What it does |
|---|---|---|
| `--artifact-dir` | *(required)* | A saved artifact dir, e.g. `artifacts/federated` or `artifacts/centralized`. |
| `--data-dir` | *(required)* | Folder of issues to score, in the same cleaned-CSV format as training. |
| `--out-csv` | `predictions.csv` | Where to write the scored rows. |
| `--batch-size` | `16` | Inference batch size. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--head-project` | `None` | Personalized artifacts only: load `heads/<name>.pt` for that project. |
| `--generic-head` | off | Personalized artifacts only: use the averaged `generic_head.pt` instead of a per-project head. |

The output CSV gains `predicted_class` (0–4) and `predicted_story_point`
(1/2/3/5/8). If the input carries a `story_point` column, metrics are printed too.

---

## Framework overview (FedSP-PEFT)

```
Each client (= one project):
  Frozen encoder (codebert-base; bert-tiny for dev)
    + LoRA adapters — FFA-LoRA: A frozen, only B trained & aggregated (exact averaging)
    + categorical embeddings (issue type, priority)
    + head:  CE   (5-logit softmax + class-weighted CrossEntropy)  [--head-type ce]
             CORN (4-logit ordinal head + corn_loss)               [--head-type corn]
  Local loss + FedProx proximal term (over aggregatable params only)
Server:
  Weighted average of aggregatable params (LoRA-B + embeddings [+ head if shared])
```

- **FFA-LoRA** (Sun et al. 2024): freezing A makes `avg(B·A) = avg(B)·A`, so
  aggregation is mathematically exact — and under 1% of parameters are
  transmitted per round.
- **FedProx / FedAvg**: `--prox-mu` above 0 is FedProx, `--prox-mu 0` is FedAvg.
- **CORN ordinal head**: decomposes the 5 classes into 4 conditional threshold
  questions, so distant misses cost more than adjacent ones.
- **Personalized heads** (FedPer/FedRep-style): the shared LoRA-B learns *what a
  complex issue looks like*; each local head learns *what this team calls an 8*.

### Conditions compared

| Condition | How to get it |
|---|---|
| Majority baseline | Always predict the most frequent class (automatic) |
| Median-SP, per project | Class nearest the project's median train SP (automatic) |
| TF-IDF + LinearSVM, per project | Classic within-project comparator (automatic) |
| Local-only | Each client trains alone, no federation (automatic) |
| Centralized | Pooled data, privacy ignored — upper bound (automatic) |
| Federated (FedAvg) | `--prox-mu 0` |
| Federated (FedProx) | `--prox-mu 1e-2`, shared head |
| Federated (FedProx + P-head) | `--personalized-head` |

The automatic ones can be turned off with `--skip-local-only`,
`--skip-centralized` and `--skip-classic-baselines`.

### Inputs & metrics

- **Text:** `title + [SEP] + description` (`</s>` for RoBERTa-family tokenizers).
- **Categorical:** issue `type` and `priority` become embeddings, fused before the head.
- **Metrics** (`results/*.json`, `summary.csv`, `summary.md`): **MAE on class
  values** and **quadratic-weighted Cohen's κ** (primary), macro-F1, accuracy,
  per-class F1, confusion matrix, and **communication cost** versus full
  fine-tuning.

> **Report per project, not pooled.** A pooled κ rewards a model merely for
> encoding project identity — the constant median predictor scores κ 0.0000 in
> all 18 projects but pools to 0.5006. Per-project models (personalized-head,
> median, TF-IDF+SVM) therefore emit no pooled entry at all.

---

## Scripts

| Script | Purpose |
|---|---|
| `export_issues.py` | TAWOS MySQL to cleaned per-project CSVs + preprocessing report |
| `train_federated_dl.py` | Main pipeline: warm-start, baselines, local-only, centralized, federated, eval |
| `run_experiments.py` | Multi-seed runner (FedProx + FedAvg per seed) |
| `compute_statistics.py` | Paired stats within a root (Wilcoxon, Friedman, Nemenyi, effect sizes) |
| `compare_conditions.py` | Paired stats **across** two experiment roots |
| `run_lopo.py` / `compare_lopo.py` | Leave-one-project-out onboarding + tidy crossover CSV |
| `predict_saved_model.py` | Inference on saved artifacts |
| `analyze_data.py` | Corpus profiling / dataset statistics |
| `sanity_check_fl_randomness.py` | Reproducibility checks, including resume bit-reproducibility |

## Project structure

```
fl/config.py            FLConfig — single source of truth for all hyperparameters
fl/data.py              Loading, filtering, per-client train/val/test split, cleaned-data validation
fl/model.py             StoryPointClassifier — frozen encoder + LoRA + embeddings + CE/CORN head
fl/client.py            FederatedClient — local training (FedProx over aggregatable params only)
fl/server.py            FedProxServer — round orchestration, per-client val, checkpoint hooks
fl/metrics.py           evaluate_classification / run_prediction (accuracy, macro-F1, MAE, kappa, CM)
fl/checkpoint.py        Save/load/rotate + RNG capture/restore for resume
fl/classic_baselines.py Per-project TF-IDF+LinearSVM and median-SP baselines
tests/                  pytest suite (preprocessing, data validation, statistics)
dashboard/              React results dashboard (see dashboard/README.md)
commands.txt            Full command runbook
```

## Notes on reproducing results

- Reported results use **`microsoft/codebert-base`**, the **CORN** head,
  **FFA-LoRA**, a **temporal** split and **3 seeds (42–44)**. The reduced seed
  count is a compute tradeoff, stated as a thesis limitation — significance comes
  from the per-project paired tests (n = 18 projects per seed).
- **Never draw accuracy conclusions from `bert-tiny` runs.** It exists to verify
  plumbing.
- The test split is evaluated exactly once per condition; validation drives all
  model selection.

## Citation

If you use the dataset, cite the TAWOS paper:

> V. Tawosi, A. Al-Subaihin, R. Moussa and F. Sarro. *A Versatile Dataset of
> Agile Open Source Software Projects.* MSR 2022.

Method references: FedAvg (McMahan et al. 2017), FedProx (Li et al. 2020), LoRA
(Hu et al. 2022), FFA-LoRA (Sun et al. 2024), CORN (Shi et al. 2021), FedPer
(Arivazhagan et al. 2019), FedRep (Collins et al. 2021), Deep-SE
(Choetkiertikul et al. 2018), GPT2SP (Fu & Tantithamthavorn 2022).
