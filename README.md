# BAFT paper results notebook

This package recalculates Tables 1–5 and Figures 1–2 of the included manuscript,
*Balance-Aware Financial Transformers for Coherent Next-Service Prediction*.
The notebook contains executed cells and visible outputs.

**Numerical correction:** the bootstrap tail comparison now handles numerical
ties consistently in the tested one-thread and eight-thread environments.
Two rounded Table 5 adjusted p-values change: accuracy 0.639 → 0.641 and FVA
0.507 → 0.508. Their conclusions remain unchanged. Original references and the
manuscript PDF are retained unmodified. See `NUMERICAL_CORRECTION.md` before
updating the manuscript or sharing revised statistical results.

The analysis uses 35 saved prediction files from seven models and five seeds,
cohort arrays, per-sequence latent measurements, recorded model settings and
timing observations. Statistics are recalculated with 5,000 user-clustered
bootstrap resamples. This evaluation does not fit models again. All 35 actual
training logs are included. Timing curves use recorded observations; runtimes
are not remeasured or simulated.

## Read the results

Extract the complete ZIP. Open `BAFT_Reproduction_Results.html` in a browser to
read the outputs without Python. Open `BAFT_Reproduction_Results.ipynb` in a
Jupyter-compatible editor to inspect the code and saved outputs.

## Execute again

Use Python 3.12 and the supplied pinned requirements. Run these commands from
the extracted folder.

Windows Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python execute_notebook.py
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python execute_notebook.py
```

The default runner uses a fresh Python/IPython process. An optional Jupyter
kernel can be used with `python execute_notebook.py --engine jupyter`. In VS
Code, select the `.venv` Python interpreter as the notebook kernel, then restart
and run all cells. Copy the package before rerunning to retain the supplied
execution. Do not combine the revised notebook with an older package folder.

## Verification and troubleshooting

SHA-256 checks verify the evidence inputs, including original and corrected
reference files. The hashing function uses chunked updates and does not require
`hashlib.file_digest`. This helper fix alone does not establish compatibility of
the pinned dependency set with Python 3.10.

Non-bootstrap-p results are checked against the original archive. Revised
bootstrap p-values are checked against the separately identified corrected
reference. All comparisons retain `rtol=atol=1e-10`; the comparison is not
bypassed. Every documented p-value correction is exported. Unanticipated
mismatches stop execution, display the affected fields, and write
`outputs/verification_mismatches.csv` plus
`outputs/runtime_environment_diagnostic.json`. Retain these files for diagnosis.

| Folder | Contents |
|---|---|
| `inputs/` | Saved predictions, cohort/latent measurements, configurations, training logs and timing observations |
| `analysis/` | Original evaluator, revised tie-aware evaluator, timing and plotting code |
| `reference_evidence/` | Unmodified archived results |
| `corrected_reference_evidence/` | Revised paired-test reference and cross-thread comparison report |
| `paper/` | Original manuscript and its expected-output inventory |
| `outputs/` | Recalculated tables, figures, logs, verification and p-value correction reports |
| `training_source/` | Original training/benchmark source for methodological inspection |
| `training/` | Actual study dataset, complete scientific training source, configuration and retraining guide |

## Dataset and independent training

The notebook now begins with **Dataset and original training code**. Its file
links open the actual study archive, preprocessing code, model definitions,
trainers and hyperparameter configuration. The preview cell reads the actual
CSV and verifies the dataset and source checksums.

The full dataset is included at
`training/data/study/mobile_money_dataset_6_month.zip`. Extract that inner ZIP
to obtain the CSV. It is the study input, not the separately labelled synthetic
software fixture. The dataset card records the reviewer access permission.

The optional independent-training cell is **disabled by default**. Enable it
only to perform 35 new fits in a separate run folder. Original training uses
Linux/WSL, Python 3.12 and the additional pinned dependencies in
`training/requirements.txt`. Detailed setup, commands and output locations are
in [training/README.md](training/README.md). New training does not overwrite the
supplied paper predictions or references. The original scientific source is
preserved byte for byte; the documented bootstrap correction is applied in a
separate analysis step.
