from exploration_utils import *
from IPython.display import display, Markdown

# ── Load and prepare data ──
df = load_adni_data()
print('Original shape:', df.shape)
df = drop_metadata(df)
print('After dropping metadata:', df.shape)
display(df.head(2))

# ── BinaryLabelDataset and split ──
dataset = make_bld(df)
dataset_train, dataset_val = split_dataset(dataset)
print(f'Train: {dataset_train.features.shape}, Val: {dataset_val.features.shape}')

# ── Baseline fairness on raw data ──
m_train = BinaryLabelDatasetMetric(dataset_train,
    unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS)
m_val = BinaryLabelDatasetMetric(dataset_val,
    unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS)
print(f'Training disparate impact = {m_train.disparate_impact():.4f}')
print(f'Validation disparate impact = {m_val.disparate_impact():.4f}')

res_svc_base_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='None')

res_svc_base_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='None')

res_svc_dir_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='DIR')

res_svc_dir_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='DIR')

res_svc_rw_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='Reweighing')

res_svc_rw_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='Reweighing')

res_svc_ad_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='AdversarialDebiasing')

res_svc_ad_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='AdversarialDebiasing')

res_svc_pr_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='PrejudiceRemover')

res_svc_pr_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='PrejudiceRemover')

res_svc_ro_smote = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=True, use_class_weight=False, debias_name='RejectOption')

res_svc_ro_cw = run_comparison(
    'svc', dataset_train, dataset_val,
    use_smote=False, use_class_weight=True, debias_name='RejectOption')

configs = [
    ('Baseline', 'SMOTE', res_svc_base_smote),
    ('Baseline', 'class_weight', res_svc_base_cw),
    ('DIR', 'SMOTE', res_svc_dir_smote),
    ('DIR', 'class_weight', res_svc_dir_cw),
    ('Reweighing', 'SMOTE', res_svc_rw_smote),
    ('Reweighing', 'class_weight', res_svc_rw_cw),
    ('AdversarialDebiasing', 'SMOTE', res_svc_ad_smote),
    ('AdversarialDebiasing', 'class_weight', res_svc_ad_cw),
    ('PrejudiceRemover', 'SMOTE', res_svc_pr_smote),
    ('PrejudiceRemover', 'class_weight', res_svc_pr_cw),
    ('RejectOption', 'SMOTE', res_svc_ro_smote),
    ('RejectOption', 'class_weight', res_svc_ro_cw),
]

rows = [results_row(d, i, r) for d, i, r in configs]
df_svc = pd.DataFrame(rows)
print('=== SVC — Complete Results ===')
display(df_svc.round(4))
save_results('SVC', df_svc)

# ── Best config by DI improvement (closest to 1.0) while BA doesn't drop below baseline ──
baseline_ba = df_svc.loc[df_svc['Debiasing'] == 'Baseline', 'BA (deb)'].max()
candidates = df_svc[df_svc['BA (deb)'] >= baseline_ba * 0.95].copy()
candidates['DI_dist'] = (candidates['DI (deb)'] - 1.0).abs()
best = candidates.loc[candidates['DI_dist'].idxmin()]
print('Best SVC configuration:')
print(best[['Debiasing', 'Imbalance', 'BA (deb)', 'DI (deb)', 'Improved']].to_string())

# ── Fairness-Accuracy scatter ──
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
for (d, i, _), c in zip(configs, colors):
    row = df_svc[(df_svc['Debiasing'] == d) & (df_svc['Imbalance'] == i)].iloc[0]
    ax.scatter(row['DI (deb)'], row['BA (deb)'], s=120, c=[c], label=f'{d}+{i}')
ax.axvline(1.0, color='gray', ls='--', alpha=0.4, label='DI=1 (parity)')
ax.set_xlabel('Disparate Impact (debiased)')
ax.set_ylabel('Balanced Accuracy (debiased)')
ax.set_title('SVC: Fairness vs Accuracy — all configurations')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
plt.tight_layout()
plt.show()
