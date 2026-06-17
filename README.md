# GenderBias_AD

Investigating gender bias in machine learning-based Alzheimer's Disease diagnosis using ADNI data.

## Background

ML models for AD diagnosis may perpetuate or amplify biases against demographic subgroups. This project evaluates gender-related fairness of a classifier trained on hippocampal volumetrics, cognitive assessments, and demographic data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

## Dataset

- **Source:** ADNI (`data_4mod_MCIvsAD.csv`)
- **Subjects:** 757 (420 male, 337 female)
- **Classes:** MCI (n=600, 79.3%) vs AD (n=157, 20.7%) — imbalanced
- **Features:** 147 columns — demographics, APOE genotype, cognitive scores (MMSE, CDR, FAQ), hippocampal subfield volumes
- **Protected attribute:** `Sex` (0=male, 1=female)
- **18 columns dropped** before training: `PTID`, `CDRSB`, `PTHAND`, `PTCOGBEG`, `DXDSEV`, `DXMDUE`, `DXMCI`, `DXAD`, `DXAPP`, `DXDDUE`, `PTADDX`, `ICV`, `DXMPTR1`–`DXMPTR6`

## Pipeline

1. **Data prep** — aif360 `BinaryLabelDataset` (label=`DIAGNOSIS`, favorable=1, protected=`Sex`)
2. **Split** — 66/34 train/validation (`dataset_Binary.split([0.66])`)
3. **Resampling** — SMOTE (`random_state=2`) on training set to balance MCI/AD (405 each)
4. **Scaling** — `StandardScaler` fit on resampled train, applied to validation
5. **Classifier** — SVC (poly kernel, C=0.1, gamma=0.1, probability=True, `random_state=42`)
6. **Debiasing** — `DisparateImpactRemover(repair_level=1)` pre-processing, then retrain SVC
7. **Threshold tuning** — scan 0.01–0.99 maximizing balanced accuracy
8. **Explanation** — LIME on 13 cases corrected by debiasing, with per-gender t-tests

## Results

### Original model (SVC)

| Metric | Value |
|---|---|
| Balanced accuracy | 0.8554 |
| Average odds difference | −0.0791 |
| Disparate impact | 0.9350 |
| Equal opportunity difference | −0.1290 |
| Statistical parity difference | −0.0267 |
| Theil index | 0.0606 |

Best threshold: **0.0298**

### DisparateImpactRemover debiased model

| Metric | Value |
|---|---|
| Balanced accuracy | 0.8976 |
| Average odds difference | −0.0553 |
| Disparate impact | 1.0591 |
| Equal opportunity difference | −0.1321 |
| Statistical parity difference | 0.0138 |
| Theil index | 0.0524 |

Best threshold: **0.4852**

Debiasing improved balanced accuracy (+4.2pp) and brought disparate impact closer to 1.0, but equal opportunity difference remained largely unchanged.

## Key findings

- **13 validation cases** were misclassified by the original model and corrected after debiasing — predominantly female AD patients who were originally misclassified as MCI (false negatives).
- After debiasing, `CDGLOBAL`, `FAQBEVG`, and APOE genotype features (`GENOTYPE_4/4`, `GENOTYPE_2/4`) became the dominant contributors.
- Among the corrected cases, `CDGLOBAL` and `FAQBEVG` were significant for both sexes; `CDCARE` was additionally significant for females.

## Files

| File | Description |
|---|---|
| `GenderBias_AD.ipynb` | Main analysis (executed, `.venv` kernel) |
| `GenderBias_AD_clean.ipynb` | Cleaned main notebook with markdown headers, inline comments, appendix |
| `GenderBias_AD_sigBias.ipynb` | Secondary 10-fold CV analysis (not executed) |
| `GenderBias_AD_sigBias_clean.ipynb` | Cleaned secondary notebook with SMOTE+scaling, fixed SVC params, consolidated dashboard |
| `data_4mod_MCIvsAD.csv` | ADNI dataset (757 × 147) |
| `AGENTS.md` | Notes for AI coding agents |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore rules |

## Dependencies

`aif360`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `lime`, `shap`, `matplotlib`, `pandas`, `numpy`, `tqdm`, `scipy`, `tensorflow`

## Running

```bash
source .venv/bin/activate
jupyter notebook
```

Open either `.ipynb` file and run all cells. The CSV is read from the same directory.
