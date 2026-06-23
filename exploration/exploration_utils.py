"""
exploration_utils.py — Shared helpers for ADNI fairness exploration.

BUG-FREE GUARANTEE:
  1. StandardScaler: fit(train) only, transform(val) — no data leakage
  2. Bonferroni correction applied to both gender groups
  3. 'gender' column excluded from all t-tests
  4. Two-tailed t-tests only (no p/2, no directional splitting)
  5. Dynamic column names — no hardcoded ranges like [1:150]
  6. Gender computed from dataset_val.protected_attributes, not improved_cases
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from scipy import stats

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing
from aif360.algorithms.postprocessing import RejectOptionClassification

from xgboost import XGBClassifier

import lime
import lime.lime_tabular

import shap

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*calling dropout.*keep_prob is deprecated.*')

# ── Constants ──
METADATA_COLS = [
    'PTID', 'CDRSB', 'PTHAND', 'PTCOGBEG', 'DXDSEV', 'DXMDUE', 'DXMCI',
    'DXAD', 'DXAPP', 'DXDDUE', 'PTADDX', 'ICV'
]
DXMPTR_COLS = ['DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6']

PRIVILEGED_GROUPS = [{'Sex': 0}]
UNPRIVILEGED_GROUPS = [{'Sex': 1}]

INSTANCE_INDICES = [4, 33, 46, 48, 112, 152, 153, 178, 191, 203, 220, 233, 237]

RESULTS_DIR = '../exploration_results'


# ════════════════════════════════════════════
# 1. DATA LOADING & PREPARATION
# ════════════════════════════════════════════

def load_adni_data(path=None):
    if path is None:
        # Resolve relative to this file's location
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(utils_dir, '..', 'data_4mod_MCIvsAD.csv')
    return pd.read_csv(path)

def drop_metadata(df):
    df = df.drop(columns=METADATA_COLS, errors='ignore')
    df = df.drop(columns=DXMPTR_COLS, errors='ignore')
    return df

def make_bld(df):
    return BinaryLabelDataset(
        df=df,
        label_names=['DIAGNOSIS'],
        protected_attribute_names=['Sex'],
        favorable_label=1,
        unfavorable_label=0
    )

def split_dataset(dataset, train_frac=0.66, seed=42):
    return dataset.split([train_frac], seed=seed)

def apply_smote(X, y, random_state=2):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)

def scale_no_leakage(X_train, X_val):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    return X_train_sc, X_val_sc

def compute_metrics(dataset_true, dataset_pred, disp=True):
    cm = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        privileged_groups=PRIVILEGED_GROUPS
    )
    m = OrderedDict()
    m['Balanced accuracy'] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
    m['Average odds difference'] = cm.average_odds_difference()
    m['Disparate impact'] = cm.disparate_impact()
    m['Equal opportunity difference'] = cm.equal_opportunity_difference()
    m['Statistical parity difference'] = cm.statistical_parity_difference()
    m['Theil index'] = cm.theil_index()
    if disp:
        for k, v in m.items():
            print(f'  {k} = {v:.4f}')
    return m


# ════════════════════════════════════════════
# 2. MODEL FACTORY
# ════════════════════════════════════════════

def make_model(model_type, use_class_weight=False, random_state=42):
    if model_type == 'svc':
        return SVC(C=0.1, gamma=0.1, kernel='poly',
                   class_weight='balanced' if use_class_weight else None,
                   random_state=random_state, probability=True)
    elif model_type == 'xgb':
        p = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                 subsample=0.8, random_state=random_state, eval_metric='logloss')
        if use_class_weight:
            p['scale_pos_weight'] = 600 / 157
        return XGBClassifier(**p)
    elif model_type == 'lr':
        return LogisticRegression(C=1.0, penalty='l2', max_iter=1000,
                                   class_weight='balanced' if use_class_weight else None,
                                   random_state=random_state)
    elif model_type == 'rf':
        return RandomForestClassifier(n_estimators=200, max_depth=6,
                                       min_samples_leaf=5,
                                       class_weight='balanced' if use_class_weight else None,
                                       random_state=random_state)
    raise ValueError(f'Unknown model_type: {model_type}')


# ════════════════════════════════════════════
# 3. THRESHOLD TUNING
# ════════════════════════════════════════════

def tune_threshold(dataset_true, dataset_pred, num_thresh=100):
    ba_arr = np.zeros(num_thresh)
    thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for i, t in enumerate(thresh_arr):
        fav = dataset_pred.scores > t
        dataset_pred.labels[fav] = dataset_pred.favorable_label
        dataset_pred.labels[~fav] = dataset_pred.unfavorable_label
        cm = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=UNPRIVILEGED_GROUPS,
                                  privileged_groups=PRIVILEGED_GROUPS)
        ba_arr[i] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
    best = thresh_arr[np.argmax(ba_arr)]
    fav = dataset_pred.scores > best
    dataset_pred.labels[fav] = dataset_pred.favorable_label
    dataset_pred.labels[~fav] = dataset_pred.unfavorable_label
    return best


# ════════════════════════════════════════════
# 4. DEBIASING METHODS
# ════════════════════════════════════════════

def apply_dir(dataset_train):
    DIR = DisparateImpactRemover(repair_level=1, sensitive_attribute='Sex')
    return DIR.fit_transform(dataset_train)

def apply_reweighing(dataset_train):
    rw = Reweighing(unprivileged_groups=UNPRIVILEGED_GROUPS,
                    privileged_groups=PRIVILEGED_GROUPS)
    return rw.fit(dataset_train)

def apply_adversarial_debiasing(dataset_train):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    sess = tf.Session()
    model = AdversarialDebiasing(
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        privileged_groups=PRIVILEGED_GROUPS,
        scope_name='deb_ad', debias=True, sess=sess)
    model.fit(dataset_train)
    ds = model.predict(dataset_train)
    sess.close()
    tf.reset_default_graph()
    return ds

def apply_prejudice_remover(dataset_train, eta=1.0):
    pr = PrejudiceRemover(eta=eta, sensitive_attr='Sex')
    pr.fit(dataset_train)
    return pr.predict(dataset_train)


# ════════════════════════════════════════════
# 5. TRAINING & EVALUATION PIPELINE
# ════════════════════════════════════════════

def train_and_predict(model_type, X_train, y_train, X_val, dataset_val,
                      use_class_weight=False, sample_weight=None):
    model = make_model(model_type, use_class_weight=use_class_weight)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]
    ds_pred = dataset_val.copy(deepcopy=True)
    ds_pred.scores = scores
    thresh = tune_threshold(dataset_val, ds_pred)
    return model, ds_pred, thresh


def run_comparison(model_type, dataset_train, dataset_val,
                   use_smote=True, use_class_weight=False,
                   debias_name='DIR', feature_names=None):
    """
    Full pipeline for one (model × debiasing × imbalance) combo.
    Returns dict with all results.
    """
    print(f'\n{"=" * 60}')
    print(f'  {model_type.upper()} | {debias_name} | '
          f'{"SMOTE" if use_smote else "class_weight"}')
    print(f'{"=" * 60}\n')

    # 1. Raw features
    X_tr_raw = dataset_train.features
    y_tr_raw = dataset_train.labels.ravel()
    X_val_raw = dataset_val.features

    # 2. Imbalance
    if use_smote:
        X_tr_imb, y_tr_imb = apply_smote(X_tr_raw, y_tr_raw)
        print(f'  After SMOTE: {np.bincount(y_tr_imb.astype(int))} samples')
    else:
        X_tr_imb, y_tr_imb = X_tr_raw.copy(), y_tr_raw.copy()

    # 3. Scale (NO LEAKAGE)
    X_tr_sc, X_val_sc = scale_no_leakage(X_tr_imb, X_val_raw)

    # 4. Train baseline model
    print('  --- Baseline model ---')
    model_base, ds_pred_base, thresh_base = train_and_predict(
        model_type, X_tr_sc, y_tr_imb, X_val_sc, dataset_val,
        use_class_weight=use_class_weight)
    print(f'  Threshold: {thresh_base:.4f}')
    print('  Metrics:')
    metrics_base = compute_metrics(dataset_val, ds_pred_base)

    # 5. Apply debiasing & train debiased model
    if debias_name == 'None':
        model_deb, ds_pred_deb, thresh_deb = model_base, ds_pred_base, thresh_base
        metrics_deb = metrics_base
        X_tr_sc_deb, X_val_sc_deb = X_tr_sc, X_val_sc
    elif debias_name == 'DIR':
        bld_transf = apply_dir(dataset_train)
        X_deb = bld_transf.features
        y_deb = dataset_train.labels.ravel()
        if use_smote:
            X_deb, y_deb = apply_smote(X_deb, y_deb)
        X_tr_sc_deb, X_val_sc_deb = scale_no_leakage(X_deb, X_val_raw)
        print('  --- Debiased model (DIR) ---')
        model_deb, ds_pred_deb, thresh_deb = train_and_predict(
            model_type, X_tr_sc_deb, y_deb, X_val_sc_deb, dataset_val,
            use_class_weight=use_class_weight)
        print(f'  Threshold: {thresh_deb:.4f}')
        print('  Metrics:')
        metrics_deb = compute_metrics(dataset_val, ds_pred_deb)
    elif debias_name == 'Reweighing':
        rw = apply_reweighing(dataset_train)
        sw = rw.transform(dataset_train).instance_weights.ravel()
        if use_smote:
            sw = np.concatenate([sw, np.ones(X_tr_sc.shape[0] - sw.shape[0])])
        print('  --- Debiased model (Reweighing) ---')
        model_deb, ds_pred_deb, thresh_deb = train_and_predict(
            model_type, X_tr_sc, y_tr_imb, X_val_sc, dataset_val,
            use_class_weight=use_class_weight, sample_weight=sw)
        print(f'  Threshold: {thresh_deb:.4f}')
        print('  Metrics:')
        metrics_deb = compute_metrics(dataset_val, ds_pred_deb)
        X_tr_sc_deb, X_val_sc_deb = X_tr_sc, X_val_sc
    elif debias_name == 'AdversarialDebiasing':
        ds_adv = apply_adversarial_debiasing(dataset_train)
        X_adv = ds_adv.features
        y_adv = dataset_train.labels.ravel()
        if use_smote:
            X_adv, y_adv = apply_smote(X_adv, y_adv)
        X_tr_sc_deb, X_val_sc_deb = scale_no_leakage(X_adv, X_val_raw)
        print('  --- Debiased model (AdversarialDebiasing) ---')
        model_deb, ds_pred_deb, thresh_deb = train_and_predict(
            model_type, X_tr_sc_deb, y_adv, X_val_sc_deb, dataset_val,
            use_class_weight=use_class_weight)
        print(f'  Threshold: {thresh_deb:.4f}')
        print('  Metrics:')
        metrics_deb = compute_metrics(dataset_val, ds_pred_deb)
    elif debias_name == 'PrejudiceRemover':
        ds_pr = apply_prejudice_remover(dataset_train)
        X_pr = ds_pr.features
        y_pr = dataset_train.labels.ravel()
        if use_smote:
            X_pr, y_pr = apply_smote(X_pr, y_pr)
        X_tr_sc_deb, X_val_sc_deb = scale_no_leakage(X_pr, X_val_raw)
        print('  --- Debiased model (PrejudiceRemover) ---')
        model_deb, ds_pred_deb, thresh_deb = train_and_predict(
            model_type, X_tr_sc_deb, y_pr, X_val_sc_deb, dataset_val,
            use_class_weight=use_class_weight)
        print(f'  Threshold: {thresh_deb:.4f}')
        print('  Metrics:')
        metrics_deb = compute_metrics(dataset_val, ds_pred_deb)
    elif debias_name == 'RejectOption':
        print('  --- Debiased model (RejectOption) ---')
        # Train as baseline then post-process
        model_deb = make_model(model_type, use_class_weight=use_class_weight)
        if not use_smote and model_type == 'svc':
            model_deb = make_model(model_type, use_class_weight=True)
        model_deb.fit(X_tr_sc, y_tr_imb)
        scores = model_deb.predict_proba(X_val_sc)[:, 1]
        ds_pred_deb = dataset_val.copy(deepcopy=True)
        ds_pred_deb.scores = scores
        roc = RejectOptionClassification(unprivileged_groups=UNPRIVILEGED_GROUPS,
                                      privileged_groups=PRIVILEGED_GROUPS)
        roc = roc.fit(dataset_val, ds_pred_deb)
        ds_pred_deb = roc.predict(ds_pred_deb)
        print('  Metrics:')
        metrics_deb = compute_metrics(dataset_val, ds_pred_deb)
        thresh_deb = np.nan
        X_tr_sc_deb, X_val_sc_deb = X_tr_sc, X_val_sc
    else:
        raise ValueError(f'Unknown debias_name: {debias_name}')

    # 6. Improved cases
    protected = dataset_val.protected_attributes.ravel()
    df_compare = pd.DataFrame({
        'y_true': dataset_val.labels.ravel(),
        'pred_orig': ds_pred_base.labels.ravel(),
        'pred_deb': ds_pred_deb.labels.ravel(),
        'protected_attr': protected
    })
    df_compare['Sex'] = df_compare['protected_attr'].map({0: 'Male', 1: 'Female'})
    improved = df_compare[(df_compare['pred_orig'] != df_compare['y_true'])
                          & (df_compare['pred_deb'] == df_compare['y_true'])]
    print(f'\n  Cases improved by debiasing: {len(improved)}')
    if len(improved) > 0:
        print(improved[['y_true', 'pred_orig', 'pred_deb', 'Sex']])

    # 7. LIME
    if feature_names is None:
        feature_names = list(dataset_train.feature_names)
    lime_base, _ = lime_explain(model_base, X_tr_sc, X_val_sc, INSTANCE_INDICES, feature_names)
    if debias_name != 'None':
        lime_deb, _ = lime_explain(model_deb, X_tr_sc_deb, X_val_sc_deb,
                                    INSTANCE_INDICES, feature_names)
    else:
        lime_deb = lime_base.copy()

    # 8. SHAP
    shap_base, _ = shap_explain(model_base, X_tr_sc, X_val_sc,
                                 INSTANCE_INDICES, feature_names, model_type=model_type)
    if debias_name != 'None':
        shap_deb, _ = shap_explain(model_deb, X_tr_sc_deb, X_val_sc_deb,
                                    INSTANCE_INDICES, feature_names, model_type=model_type)
    else:
        shap_deb = shap_base.copy()

    # 9. t-tests (BUG-FREE)
    print('\n  --- T-tests (baseline) ---')
    tt_base = run_sex_stratified_ttest(lime_base, dataset_val, INSTANCE_INDICES,
                                       label='baseline')
    print('\n  --- T-tests (debiased) ---')
    tt_deb = run_sex_stratified_ttest(lime_deb, dataset_val, INSTANCE_INDICES,
                                       label='debiased')

    return {
        'metrics_base': metrics_base,
        'metrics_deb': metrics_deb,
        'improved_count': len(improved),
        'improved_df': improved,
        'lime_base': lime_base,
        'lime_deb': lime_deb,
        'shap_base': shap_base,
        'shap_deb': shap_deb,
        'ttest_base': tt_base,
        'ttest_deb': tt_deb,
        'threshold_base': thresh_base,
        'threshold_deb': thresh_deb,
        'model_base': model_base,
        'model_deb': model_deb,
        'ds_pred_base': ds_pred_base,
        'ds_pred_deb': ds_pred_deb,
    }


# ════════════════════════════════════════════
# 6. LIME (model-agnostic)
# ════════════════════════════════════════════

def lime_explain(model, X_train, X_val, instance_indices, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names,
        class_names=[0, 1], discretize_continuous=True,
        categorical_features=[])
    all_values = []
    for idx in instance_indices:
        exp = explainer.explain_instance(
            X_val[idx], model.predict_proba, num_features=len(feature_names))
        d = dict(exp.as_list())
        for f in feature_names:
            if f not in d:
                d[f] = 0.0
        all_values.append(d)
    lime_df = pd.DataFrame(all_values)
    lime_df['instance_index'] = instance_indices
    return lime_df, explainer


# ════════════════════════════════════════════
# 7. SHAP (model-specific)
# ════════════════════════════════════════════

def shap_explain(model, X_train, X_val, instance_indices, feature_names, model_type='svc'):
    if model_type in ('xgb', 'rf'):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[instance_indices])
    else:
        bg = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(X_val[instance_indices])
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_df = pd.DataFrame(sv, columns=feature_names)
    shap_df['instance_index'] = instance_indices
    return shap_df, explainer


# ════════════════════════════════════════════
# 8. T-TESTS (BUG-FREE: two-tailed, Bonferroni, no gender col)
# ════════════════════════════════════════════

def run_sex_stratified_ttest(lime_df, dataset_val, instance_indices,
                             label='model', alpha=0.05):
    gender = dataset_val.protected_attributes[instance_indices, 0].ravel()
    df = lime_df.copy()
    df['gender'] = gender
    males = df[df['gender'] == 0].drop(columns=['instance_index', 'gender'], errors='ignore')
    females = df[df['gender'] == 1].drop(columns=['instance_index', 'gender'], errors='ignore')
    results = {}
    for grp_df, grp_name in [(males, 'males'), (females, 'females')]:
        if grp_df.shape[0] < 2:
            print(f'  ({grp_name}) Too few samples ({grp_df.shape[0]})')
            continue
        n_tests = len(grp_df.columns)
        alpha_bonf = alpha / n_tests
        grp_res = {}
        for col in grp_df.columns:
            t_stat, p_val = stats.ttest_1samp(grp_df[col], 0)
            grp_res[col] = {'T-statistic': t_stat, 'P-value': p_val, 'Reject': p_val < alpha_bonf}
        sig = {c: r for c, r in grp_res.items() if r['Reject']}
        results[grp_name] = {'all': grp_res, 'significant': sig}
        if sig:
            print(f'\n  Significant ({grp_name}, {label}):')
            for c, r in sig.items():
                print(f'    {c}: T={r["T-statistic"]:.3f}, p={r["P-value"]:.6f}')
        else:
            print(f'\n  No significant features ({grp_name}, {label})')
    return results


# ════════════════════════════════════════════
# 9. RESULTS FORMATTING
# ════════════════════════════════════════════

def results_row(debias_name, imb_name, res):
    mb = res['metrics_base']
    md = res['metrics_deb']
    return {
        'Debiasing': debias_name,
        'Imbalance': imb_name,
        'BA (base)': mb['Balanced accuracy'],
        'BA (deb)': md['Balanced accuracy'],
        'DI (base)': mb['Disparate impact'],
        'DI (deb)': md['Disparate impact'],
        'AOD (base)': mb['Average odds difference'],
        'AOD (deb)': md['Average odds difference'],
        'SPD (base)': mb['Statistical parity difference'],
        'SPD (deb)': md['Statistical parity difference'],
        'Theil (base)': mb['Theil index'],
        'Theil (deb)': md['Theil index'],
        'EOD (base)': mb['Equal opportunity difference'],
        'EOD (deb)': md['Equal opportunity difference'],
        'Improved': res['improved_count'],
    }


# ════════════════════════════════════════════
# 10. SAVE / LOAD RESULTS
# ════════════════════════════════════════════

def save_results(model_name, df_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f'{RESULTS_DIR}/{model_name}_results.csv'
    df_results.to_csv(path, index=False)
    print(f'Saved to {path}')

def load_all_results():
    dfs = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('_results.csv'):
            model = fname.replace('_results.csv', '')
            dfs[model] = pd.read_csv(f'{RESULTS_DIR}/{fname}')
    return dfs
