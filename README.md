# BAFT: Balance-Aware Financial Transformer

Research code for mobile-money next-service prediction and candidate-specific
prediction of transaction direction, amount, fee, and resulting balance.
BAFT combines a causal sequence encoder with latent-continuity regularization
and soft accounting and affordability controls. Here, *causal* means that an
encoder position attends only to the available input prefix; it does not denote
causal inference.

This reviewer distribution contains the data preparation, model training,
ablations, statistical analysis, and computational benchmarks associated with
the study, together with the actual historical study input archive:

- [Study dataset: mobile_money_dataset_6_month.zip](data/study/mobile_money_dataset_6_month.zip) — 3,134,000 transaction rows from 30,000 users.
- [Dataset card](data/DATASET_CARD.md) — provenance, contents, and reviewer access scope.
- [Dataset manifest](data/dataset_manifest.json) — archive checksums and dataset inventory.

The study archive's SHA-256 is
`72f455dbf235a1bc6958bee5a9a1dea9b5facc89853844e5f82895c63c03070a`,
matching the historical input hash recorded in the supplied experiment evidence.
Frozen aggregate reference results and an independently generated synthetic
software fixture are also included. Historical fitted checkpoints and
event-level model outputs are not included.

The author reports that the data derive from actual company transactions and
that the company has permitted sharing the dataset with reviewers. This
confirmation covers reviewer access; public dataset redistribution is outside
the confirmed permission. If GitHub is used to deliver this dataset-containing copy, the repository must
remain private and access should be limited to the authorized reviewers. The
included dataset files are explicitly retained by this copy's `.gitignore`.
Use the separate code-only distribution for any public repository.

## Experimental scope

The reported study used 3,134,000 transactions from 30,000 users. Sequence
construction produced 56,357 candidate histories, of which 36,126 had eligible
Gold targets. Each history contains 100 events, followed by one target in
reconstructed event order. The primary chronological test cohort contains 8,850
sequences from 5,253 users. Equal-timestamp ordering and cohort-selection details
are documented in [Methods](docs/METHODS.md); event-order precedence does not
necessarily imply a strictly later target timestamp.

Seven configurations were evaluated over five independent training seeds:

| Configuration | Encoder and controls |
| --- | --- |
| Aggregate GBDT | Gradient-boosted trees on aggregated history features |
| GRU multitask | Recurrent encoder with candidate-specific outputs |
| Transformer | Causal attention with candidate-specific outputs |
| Time-aware Transformer | Transformer with learned relative-time bias |
| BAFT latent | Time-aware Transformer with smoothness and transition-bound penalties |
| BAFT accounting | Time-aware Transformer with accounting decoding and feasibility training |
| BAFT full | Both latent and accounting/feasibility components |

All neural comparisons use the same candidate-specific output interface. The
accounting ablation includes feasibility training; it does not isolate one
accounting equation. The reference configuration is defined in
[configs/paper.json](configs/paper.json).

## Reported findings

Across the five seeds, BAFT full attained 54.60% next-service accuracy, compared
with 54.66% for the Transformer. A separate one-sided test supported
noninferiority within a one-percentage-point margin (p = 0.002177). BAFT reduced
native operational accounting-residual MAE by 93.5%, balance MAE by 66.6%, and
measured latent-acceleration MSE by 91.3%. The corresponding user-cluster tests
had Holm-adjusted p = 0.001200.

These findings do not establish superior classification accuracy or financially
valid accuracy. Aggregate GBDT achieved the highest accuracy in the comparison.
BAFT required more computation than the Transformer. The estimates, uncertainty
intervals, test definitions, and measured hardware costs are given in
[Results](docs/RESULTS.md). “Native” in these headline comparisons refers to
financial outputs after validation-selected service gating, before hard balance
projection.

## Installation

The reference implementation uses Python 3.12 on Linux and runs on CPU. The
pinned numerical dependencies are listed in [requirements.txt](requirements.txt).
The following installs the CPU build of PyTorch before the remaining packages:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The original benchmark used an AMD EPYC 9V74 CPU with eight configured threads.
Runtime depends on hardware, library builds, and concurrent workloads. GPU and
Windows execution have not been validated; the benchmark utilities use Linux
process-memory measurements.

## Dataset inspection and software verification

Inspect the included dataset inventory and integrity information with the
Python standard library:

```bash
python scripts/inspect_datasets.py --scan-rows
```

The aggregate reference tables can be checked with the Python standard library:

```bash
python scripts/check_reference_results.py
```

After installing the dependencies, computational tests can be run with:

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```

A small synthetic run exercises cleaning, sequence construction, training,
prediction regeneration, latent evaluation, and evidence finalization. The
fixture is already included as an [open CSV](data/synthetic/synthetic_mobile_money.csv)
and [pipeline ZIP](data/synthetic/synthetic_mobile_money.zip):

```bash
python scripts/run_experiments.py --config configs/smoke.json --input data/synthetic/synthetic_mobile_money.zip --output-dir runs/smoke
```

This fixture contains entirely generated transactions. Its one-epoch runs and
50 bootstrap iterations test software execution only and cannot substantiate
the manuscript's empirical claims. The study uses eight epochs and 5,000
bootstrap iterations. [Verification](docs/VERIFICATION.md) records the checks
performed for this distribution.

## Reproducing the study

The paper configuration can use the included study archive. The expected
schema and data conditions are described in [data/README.md](data/README.md),
with provenance and reviewer access scope in the [dataset card](data/DATASET_CARD.md).
The first command previews the computational plan without running the study:

```bash
python scripts/run_experiments.py --config configs/paper.json --input data/study/mobile_money_dataset_6_month.zip --output-dir runs/paper --dry-run
python scripts/run_experiments.py --config configs/paper.json --input data/study/mobile_money_dataset_6_month.zip --output-dir runs/paper
python scripts/run_experiments.py --config configs/paper.json --output-dir runs/paper --stages inference-benchmark training-benchmark
```

The runner records commands, configurations, checksums, logs, and completion
status. It writes newly generated evidence under the selected run directory;
the frozen reference tables are preserved separately. Stage-specific commands,
restart behavior, and output locations are documented in
[Reproducing the pipeline](docs/REPRODUCING.md).

## Repository contents

| Location | Content |
| --- | --- |
| `src/baft/` | Cleaning, cache construction, model definitions, training, evaluation, and benchmark code |
| `configs/` | Study protocol and explicitly separate synthetic-check configuration |
| `scripts/run_experiments.py` | Portable stage launcher and execution records |
| `scripts/make_synthetic_fixture.py` | Synthetic input generator |
| `scripts/inspect_datasets.py` | Standard-library dataset inventory and integrity inspection |
| `scripts/check_reference_results.py` | Aggregate consistency checks and report-table export |
| `tests/` | Synthetic tests of model semantics and evidence aggregation |
| `results/reference/` | Aggregate study metrics, paired effects, and compute reports |
| `docs/` | Implemented methods, reproduction instructions, results, provenance, and verification |
| `data/study/mobile_money_dataset_6_month.zip` | Actual historical study input archive included for reviewer access |
| `data/synthetic/` | Independently generated synthetic CSV and matching ZIP |
| `data/README.md`, `data/DATASET_CARD.md`, `data/dataset_manifest.json` | Input schema, provenance, access scope, and dataset integrity records |

## Interpretation and reproducibility limits

The financial controls are soft: they can reduce inconsistency without
guaranteeing affordability or exact accounting identities. Hard projection
repairs the predicted balance by construction but leaves service ranking and
debit affordability unchanged. Latent continuity is a modeling assumption,
and reduced latent acceleration is an internal measurement rather than direct
proof of more realistic behavior.

The primary evaluation is conditional on Gold-target eligibility. Users may
appear in more than one chronological split. User-cluster intervals account
for repeated users in the test cohort while conditioning on the fitted models;
seed variability is reported separately. Checking the supplied aggregate tables
does not independently validate the underlying data or recompute cluster
resampling. The included transaction archive enables inspection of the input
and reconstruction of the cohort. Recomputing historical cluster resampling
also requires the original prediction and latent event outputs, which are not
included. Running the pipeline produces new fitted models and outputs; matching
seeds alone does not guarantee identical results across numerical environments.

## Attribution and licensing

This distribution supplies computational material for the accompanying BAFT
manuscript. It does not assert a publication venue, acceptance status, DOI,
author affiliation, or software license. Bibliographic metadata and a license
may be added by the rights holders. The author-reported permission to share
the dataset with reviewers does not establish permission for public dataset
redistribution.
