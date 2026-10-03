# Experimental results

Raw result files for the paper *Story Point Estimation Across Agile Projects with
Personalized Federated Learning: An Empirical Study*. Results are grouped by
research question. Every run uses 18 TAWOS projects (one project = one client),
a per-client temporal split, CodeBERT-base, a CORN ordinal head, 60 rounds with
1 local epoch, and seeds 42, 43, and 44.

## Layout

```
RQ1/  Federated vs. local vs. centralized
  shared_head_runs/seed_<N>/fedprox/   FedSP-PEFT (shared head, FedProx mu=0.01) plus the
                                       baselines run with it: local-only, centralized,
                                       TF-IDF+SVM, per-project median
  shared_head_runs/seed_<N>/fedavg/    Shared head with mu=0 (FedAvg)
  statistics/<metric>/                 Friedman, Nemenyi, pairwise Wilcoxon and Cliff's
                                       delta across the six RQ1 conditions
  supporting_mu_0.001_runs/            mu=0.001 sensitivity runs (seeds 42 and 43 only)

RQ2/  Impact of personalization
  personalized_head_runs/seed_<N>/fedprox/   FedSP-PEFT-P (per-client heads)
  comparison_shared_vs_personalized/<metric>/  paired comparison against the RQ1
                                               shared-head FedProx runs

RQ3/  Parameter-efficiency trade-off
  full_finetuning_runs/seed_<N>/fedprox/     full-model fine-tuning (LoRA disabled)
  comparison_lora_vs_full_finetuning/<metric>/  paired comparison against the RQ1
                                                shared-head FedProx runs (FFA-LoRA)

test_split/test_split.csv   The per-client test issues. The temporal split does not
                            depend on the seed, so this file is identical for every run.
```

`<metric>` is `mae`, `macro_f1`, or `cohen_kappa` (quadratic-weighted kappa).

The shared-head FedProx runs in `RQ1/shared_head_runs/*/fedprox/` are the reference
condition for RQ2 and RQ3 and are stored only once.

## Files in a run folder

| File | Content |
|---|---|
| `config.json` | Full configuration of the run |
| `federated_per_project.json` | Per-project test metrics of the federated model: accuracy, macro-F1, per-class F1, confusion matrix, MAE, QWK |
| `local_only_per_project.json`, `centralized_per_project.json` | Same metrics for the local-only and centralized models (RQ1 `fedprox` folders only) |
| `tfidf_svm_per_project.json`, `median_per_project.json` | Same metrics for the two simple baselines (RQ1 `fedprox` folders only) |
| `federated_round_history.json` | Per-round training loss and validation metrics |
| `communication_cost.json` | Trainable-parameter count and estimated upload volume |
| `summary.csv`, `summary.md` | Per-project summary table of the run |

## Files in a statistics or comparison folder

| File | Content |
|---|---|
| `results_long.csv` / `paired_values.csv` | Per-(seed, project) values used by the tests |
| `pairwise_vs_fedprox.csv` / `summary.json` | Wilcoxon signed-rank statistic and p-value, Vargha-Delaney A12, Cliff's delta, win counts |
| `friedman.json`, `nemenyi.csv` | Friedman test, average ranks, and Nemenyi post-hoc p-values (RQ1) |
| `summary_table.tex` | LaTeX summary table (RQ1) |
| `plot_*.py`, `results_rq1_*.pdf/png` | Scripts and output for the per-project MAE and critical-difference figures (RQ1, `mae`) |

## Not included

Model weights, checkpoints, tokenizer files, and training logs are not included
because of their size. All reported numbers can be recomputed from the JSON files here.
