# Pundit Deculs - IDSSE v2

Code, frozen model, dependencies, and reports for the v2 candidate.

Challenge data is not included. For the methodology, see `Pundit_Deculs.md`.

## Installation and Inference

Use the Python version specified in `ENVIRONMENT.json`. Create the virtual environment outside the code directory:

```cmd
python -m venv "E:\MTAH2027\.venv_reproduce_v2"
call "E:\MTAH2027\.venv_reproduce_v2\Scripts\activate.bat"
python -m pip install -r requirements.txt
python pundit_v2.py candidate --data-dir "E:\More Than A Hack 2027 - Initial Challenge - 13"
python evaluation\validate_submission.py evaluation\candidates\idsse_v2\Pundit_Deculs.csv
certutil -hashfile evaluation\candidates\idsse_v2\Pundit_Deculs.csv SHA256
```

The expected hash is recorded in `EXPECTED_SUBMISSION.sha256`.

Candidate mode uses the frozen model without retraining or accessing ground-truth data (GT).

Only load joblib model weights from trusted sources.

## Reproducing External Experiments

Previously completed reports are reused as-is. To run new experiments, use a separate output directory:

```cmd
python pundit_v2.py develop --out evaluation\experiments\idsse_v2_retrained
python pundit_v2.py test --out evaluation\experiments\idsse_v2_retrained
python pundit_v2.py diagnose --out evaluation\experiments\idsse_v2_retrained
```

The development process downloads publicly available IDSSE XML data into `data/external/idsse`.

The v1 model weights are included as an external baseline.

Models produced by experiments in separate directories will not overwrite the submission model.

Do not select or tune a model based on challenge ground-truth data (GT).

## Contents and Usage Rights

- **Metrica upstream:** MIT License. See `LICENSE-METRICA`.
- **External data:** CC BY 4.0. See `DATA_SOURCES.md`.
- **Team code:** MIT License. See `LICENSE-TEAM`.
- **Challenge materials:** Videos, ATD, tagging data, homography data, derived challenge coordinates, and ground truth (GT) are not included.

`.gitattributes` preserves file bytes during Git checkout.

`.gitignore` excludes local data and generated outputs.

This package has not yet been published or submitted using the packaging utility.
