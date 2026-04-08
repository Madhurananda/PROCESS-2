#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_data_analysis.py

Generate descriptive statistics and publication‑ready figures for the PROCESS‑2 dataset:
- Demographic summary tables (age, gender, MMSE) by diagnosis and train/test split
- Age distribution (raincloud plots) with Kruskal‑Wallis and post‑hoc Dunn tests
- MMSE distribution (raincloud plots) with statistical tests
- Gender distribution (stacked bar charts) with chi‑square tests
- Audio duration and SNR analysis (raincloud plots) per task and diagnosis
- File size statistics (optional)

All figures are saved in the specified results directory.

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

# Statistical tests
from scipy.stats import shapiro, f_oneway, kruskal, chi2_contingency
import scikit_posthocs as sp

# Plotting libraries
import ptitprince as pt
from statannotations.Annotator import Annotator

# -----------------------------------------------------------------------------
# 1. Configuration and Data Loading
# -----------------------------------------------------------------------------

data_dir = '../data/'
results_dir = data_dir + '../results/FIGS/'
os.makedirs(results_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(data_dir + 'meta-info.csv')

# -----------------------------------------------------------------------------
# 2. Demographic Summary Tables (LaTeX output)
# -----------------------------------------------------------------------------

# Clean gender column
df['gender'] = df['gender'].str.lower()

# Summary by Split and diagnosis
summary = df.groupby(['Split', 'diagnosis']).agg(
    N=('IDs', 'count'),
    Age_mean=('age', 'mean'),
    Age_std=('age', 'std'),
    MMSE_N=('MMSE', 'count'),
    MMSE_mean=('MMSE', 'mean'),
    MMSE_std=('MMSE', 'std'),
    Male=('gender', lambda x: (x == 'male').sum()),
    Female=('gender', lambda x: (x == 'female').sum())
)
summary['Male_%'] = (summary['Male'] / summary['N'] * 100).round(1)
summary['Female_%'] = (summary['Female'] / summary['N'] * 100).round(1)
summary = summary.round(2)

# Total summary by diagnosis only
total = df.groupby('diagnosis').agg(
    N=('IDs', 'count'),
    Age_mean=('age', 'mean'),
    Age_std=('age', 'std'),
    MMSE_N=('MMSE', 'count'),
    MMSE_mean=('MMSE', 'mean'),
    MMSE_std=('MMSE', 'std'),
    Male=('gender', lambda x: (x == 'male').sum()),
    Female=('gender', lambda x: (x == 'female').sum())
)
total['Male_%'] = (total['Male'] / total['N'] * 100).round(1)
total['Female_%'] = (total['Female'] / total['N'] * 100).round(1)
total = total.round(2)

print("\n=== Summary by Split and Diagnosis (LaTeX) ===\n")
print(summary.to_latex)
print("\n=== Total by Diagnosis (LaTeX) ===\n")
print(total.to_latex())

# -----------------------------------------------------------------------------
# 3. Age Distribution Plots and Statistics
# -----------------------------------------------------------------------------

print("\n--- Age Analysis ---")

# Order diagnoses
diag_order = ["Dementia", "MCI", "HC"]
df['diagnosis'] = pd.Categorical(df['diagnosis'], categories=diag_order, ordered=True)

# Colors
diag_colors = ["#d32f2f", "#1976d2", "#388e3c"]
split_palette = {"TRAIN": "#0097a7", "TEST": "#f57c00"}

# Raincloud plots: overall and split
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

diag_pairs = [("Dementia", "MCI"), ("Dementia", "HC"), ("MCI", "HC")]
split_pairs = [
    (("Dementia", "TRAIN"), ("Dementia", "TEST")),
    (("MCI", "TRAIN"), ("MCI", "TEST")),
    (("HC", "TRAIN"), ("HC", "TEST"))
]

# Plot A: Overall
ax = axes[0]
pt.RainCloud(x='diagnosis', y='age', data=df, order=diag_order,
             palette=diag_colors, bw=0.2, width_viol=0.6, orient='v', alpha=0.6, ax=ax)
annotator = Annotator(ax, diag_pairs, data=df, x='diagnosis', y='age', order=diag_order)
annotator.configure(test='Kruskal', comparisons_correction='bonferroni',
                    text_format='star', loc='outside',
                    pvalue_thresholds=[(0.001, "***"), (0.01, "**"), (1.0, "ns")])
annotator.apply_and_annotate()
ax.set_title("A. Overall Age Distribution", fontweight='heavy', fontsize=16, pad=55)

# Plot B: Split
ax = axes[1]
pt.RainCloud(x='diagnosis', y='age', hue='Split', data=df, order=diag_order,
             palette=split_palette, bw=0.2, width_viol=0.6, orient='v', alpha=0.6, ax=ax)
annotator = Annotator(ax, split_pairs, data=df, x='diagnosis', y='age',
                      hue='Split', order=diag_order, hue_order=["TRAIN", "TEST"])
annotator.configure(test='Kruskal', comparisons_correction='bonferroni',
                    text_format='star', loc='outside',
                    pvalue_thresholds=[(0.001, "***"), (0.01, "**"), (1.0, "ns")])
annotator.apply_and_annotate()
ax.set_title("B. Age Balance (Train vs. Test)", fontweight='heavy', fontsize=16, pad=55)

# Formatting
for ax in axes:
    ax.set_ylabel("Age (Years)", fontweight='bold')
    ax.set_xlabel("Diagnosis", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)

# Legends
handles_diag = [mpatches.Patch(color=diag_colors[i], label=diag_order[i], alpha=0.8)
                for i in range(len(diag_order))]
axes[0].legend(handles=handles_diag, loc='upper center', bbox_to_anchor=(0.5, -0.25),
               ncol=3, frameon=True, edgecolor='black', prop={'weight': 'bold'})

handles_split = [mpatches.Patch(color=split_palette[k], label=k, alpha=0.8)
                 for k in ["TRAIN", "TEST"]]
axes[1].legend(handles=handles_split, loc='upper center', bbox_to_anchor=(0.5, -0.25),
               ncol=2, frameon=True, edgecolor='black', prop={'weight': 'bold'})

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig(results_dir + "age_distribution_comprehensive_dark.png", dpi=600, bbox_inches='tight')
plt.show()

# Statistical tests for age
print("\nAge normality (Shapiro–Wilk):")
for d in df['diagnosis'].unique():
    vals = df[df.diagnosis == d]['age']
    stat, p = shapiro(vals)
    print(f"{d}: W={stat:.3f}, p={p:.3e}")

print("\nAge Kruskal–Wallis test:")
stat, p = kruskal(df[df.diagnosis=="HC"].age,
                  df[df.diagnosis=="MCI"].age,
                  df[df.diagnosis=="Dementia"].age)
print(f"H = {stat:.3f}, p = {p:.3e}")

# -----------------------------------------------------------------------------
# 4. MMSE Distribution Plots and Statistics
# -----------------------------------------------------------------------------

print("\n--- MMSE Analysis ---")

df_mmse = df.dropna(subset=['MMSE']).copy()
df_mmse['diagnosis'] = pd.Categorical(df_mmse['diagnosis'], categories=diag_order, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot A: Overall
ax = axes[0]
pt.RainCloud(x='diagnosis', y='MMSE', data=df_mmse, order=diag_order,
             palette=diag_colors, bw=0.2, width_viol=0.6, orient='v', alpha=0.6, ax=ax)
annotator = Annotator(ax, diag_pairs, data=df_mmse, x='diagnosis', y='MMSE', order=diag_order)
annotator.configure(test='Kruskal', comparisons_correction='bonferroni',
                    text_format='star', loc='outside',
                    pvalue_thresholds=[(0.001, "***"), (0.01, "**"), (1.0, "ns")])
annotator.apply_and_annotate()
ax.set_title("A. Overall MMSE Distribution", fontweight='heavy', fontsize=16, pad=30)

# Plot B: Split
ax = axes[1]
pt.RainCloud(x='diagnosis', y='MMSE', hue='Split', data=df_mmse, order=diag_order,
             palette=split_palette, bw=0.2, width_viol=0.6, orient='v', alpha=0.6, ax=ax)
annotator = Annotator(ax, split_pairs, data=df_mmse, x='diagnosis', y='MMSE',
                      hue='Split', order=diag_order, hue_order=["TRAIN", "TEST"])
annotator.configure(test='Kruskal', comparisons_correction='bonferroni',
                    text_format='star', loc='outside',
                    pvalue_thresholds=[(0.001, "***"), (0.01, "**"), (1.0, "ns")])
annotator.apply_and_annotate()
ax.set_title("B. MMSE Balance (Train vs. Test)", fontweight='heavy', fontsize=16, pad=30)

# Dynamic y‑limits for stars
mmse_min, mmse_max = df_mmse['MMSE'].min(), df_mmse['MMSE'].max()
y_range = mmse_max - mmse_min
y_lim_min = mmse_min - 0.05 * y_range
y_lim_max = mmse_max + 0.25 * y_range
for ax in axes:
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_ylabel("MMSE Score", fontweight='bold')
    ax.set_xlabel("Diagnosis", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)

# Legends
axes[0].legend(handles=handles_diag, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               ncol=3, frameon=True, edgecolor='black', prop={'weight': 'bold'})
axes[1].legend(handles=handles_split, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               ncol=2, frameon=True, edgecolor='black', prop={'weight': 'bold'})

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig(results_dir + "MMSE_stats_dynamic_margin.png", dpi=600, bbox_inches='tight')
plt.show()

# MMSE statistical tests
print("\nMMSE normality (Shapiro–Wilk):")
for d in df_mmse['diagnosis'].unique():
    vals = df_mmse[df_mmse.diagnosis == d]['MMSE']
    stat, p = shapiro(vals)
    print(f"{d}: W={stat:.3f}, p={p:.3e}")

print("\nMMSE Kruskal–Wallis test:")
stat, p = kruskal(df_mmse[df_mmse.diagnosis=="HC"].MMSE,
                  df_mmse[df_mmse.diagnosis=="MCI"].MMSE,
                  df_mmse[df_mmse.diagnosis=="Dementia"].MMSE)
print(f"H = {stat:.3f}, p = {p:.3e}")

# Post‑hoc Dunn test for MMSE
dunn_mmse = sp.posthoc_dunn(df_mmse, val_col='MMSE', group_col='diagnosis', p_adjust='bonferroni')
print("\nDunn post‑hoc (MMSE):\n", dunn_mmse)

# -----------------------------------------------------------------------------
# 5. Gender Distribution Plots and Statistics
# -----------------------------------------------------------------------------

print("\n--- Gender Analysis ---")

# Reload original df (gender already lowercased)
df_gender = pd.read_csv(data_dir + 'meta-info.csv')
df_gender['gender'] = df_gender['gender'].str.lower()

colors_gender = ['#1F4E79', '#C65911']  # dark blue, orange
diag_order_gender = ['Dementia', 'MCI', 'HC']

def get_pct(df, group_cols):
    counts = df.groupby(group_cols + ['gender']).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return pct.reindex(diag_order_gender, level='diagnosis').reset_index()

df_overall = get_pct(df_gender, ['diagnosis'])
df_split = get_pct(df_gender, ['Split', 'diagnosis'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
plot_configs = [
    {'ax': axes[0], 'data': df_overall, 'title': 'A. Total Cohort'},
    {'ax': axes[1], 'data': df_split[df_split['Split'] == 'TRAIN'], 'title': 'B. TRAIN Split'},
    {'ax': axes[2], 'data': df_split[df_split['Split'] == 'TEST'], 'title': 'C. TEST Split'}
]

for cfg in plot_configs:
    ax = cfg['ax']
    df_plot = cfg['data'].set_index('diagnosis')[['male', 'female']]
    df_plot.plot(kind='bar', stacked=True, ax=ax, color=colors_gender,
                 width=0.7, edgecolor='white', legend=False)
    ax.set_title(cfg['title'], fontweight='bold', pad=15, fontsize=13)
    ax.set_xticklabels(diag_order_gender, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for container in ax.containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in container]
        ax.bar_label(container, labels=labels, label_type='center',
                     color='white', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    sns.despine(ax=ax)

axes[0].set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
leg = fig.legend(['Male', 'Female'], loc='upper center', bbox_to_anchor=(0.5, 0.12),
                 ncol=2, frameon=True, edgecolor='black', fontsize=12,
                 prop={'weight': 'bold', 'size': 12})
leg.get_title().set_weight('bold')
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(results_dir + 'gender_distribution_final_boxed.png', dpi=600, bbox_inches='tight')
plt.show()

# Chi‑square tests
cont_table = pd.crosstab(df_gender['diagnosis'], df_gender['gender'])
chi2, p, dof, expected = chi2_contingency(cont_table)
n = cont_table.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(cont_table.shape) - 1)))
print(f"\nGender × diagnosis: χ²={chi2:.2f}, p={p:.3e}, Cramér's V={cramers_v:.3f}")

cont_split = pd.crosstab(df_gender['Split'], df_gender['gender'])
chi2_split, p_split, _, _ = chi2_contingency(cont_split)
print(f"Gender × split: χ²={chi2_split:.2f}, p={p_split:.3e}")

# -----------------------------------------------------------------------------
# 6. Audio Duration and SNR Analysis
# -----------------------------------------------------------------------------

print("\n--- Audio Characteristics ---")

# Load audio duration and SNR data
audio_df = pd.read_csv(data_dir + 'Audio_duration.csv')
df_meta = pd.read_csv(data_dir + 'meta-info_FINAL.csv')

# Extract mean SNR from "mean ± std" string
df_meta["snr_mean"] = df_meta["AUDIO_SNR"].str.split("±").str[0].astype(float)
audio_df = audio_df.merge(df_meta[['dir_name', 'snr_mean']],
                          left_on='participant', right_on='dir_name', how='left')
audio_df.drop(columns=['dir_name'], inplace=True)

# Set categorical orders
diag_order_audio = ["Dementia", "MCI", "HC"]
task_order_audio = ["CTD", "PFT", "SFT"]   # for display order
audio_df['diagnosis'] = pd.Categorical(audio_df['diagnosis'], categories=diag_order_audio, ordered=True)
audio_df['task'] = pd.Categorical(audio_df['task'], categories=task_order_audio, ordered=True)

# Task‑level summary
task_stats = audio_df.groupby("task").agg(
    participants=("participant", "nunique"),
    recordings=("task", "count"),
    duration_mean=("duration_sec", "mean"),
    duration_std=("duration_sec", "std"),
    snr_mean=("snr_mean", "mean"),
    snr_std=("snr_mean", "std")
).round(2)
task_stats["Duration (s)"] = task_stats["duration_mean"].astype(str) + " ± " + task_stats["duration_std"].astype(str)
task_stats["SNR (dB)"] = task_stats["snr_mean"].astype(str) + " ± " + task_stats["snr_std"].astype(str)
task_stats = task_stats.drop(columns=["duration_mean", "duration_std", "snr_mean", "snr_std"])
print("\nTask summary:\n", task_stats)

# Diagnosis × task summary
task_diag_stats = audio_df.groupby(["task", "diagnosis"]).agg(
    participants=("participant", "nunique"),
    recordings=("task", "count"),
    duration_mean=("duration_sec", "mean"),
    duration_std=("duration_sec", "std"),
    snr_mean=("snr_mean", "mean"),
    snr_std=("snr_mean", "std")
).round(2)
task_diag_stats["Duration (s)"] = task_diag_stats["duration_mean"].astype(str) + " ± " + task_diag_stats["duration_std"].astype(str)
task_diag_stats["SNR (dB)"] = task_diag_stats["snr_mean"].astype(str) + " ± " + task_diag_stats["snr_std"].astype(str)
task_diag_stats = task_diag_stats.drop(columns=["duration_mean", "duration_std", "snr_mean", "snr_std"]).reset_index()
task_diag_stats.to_latex(caption="Audio characteristics by task and diagnosis", label="tab:audio_overview")

# -----------------------------------------------------------------------------
# 7. Comprehensive Audio Raincloud Plots (4 rows × 3 tasks)
# -----------------------------------------------------------------------------

print("\nGenerating audio plots...")

task_order_plot = ["SFT", "PFT", "CTD"]   # plot order
diag_order_plot = ["Dementia", "MCI", "HC"]
diag_pairs_plot = [("Dementia", "MCI"), ("Dementia", "HC"), ("MCI", "HC")]
split_pairs_plot = [
    (("Dementia", "TRAIN"), ("Dementia", "TEST")),
    (("MCI", "TRAIN"), ("MCI", "TEST")),
    (("HC", "TRAIN"), ("HC", "TEST"))
]

fig, axes = plt.subplots(4, 3, figsize=(12, 55))

for col, task in enumerate(task_order_plot):
    df_task = audio_df[audio_df["task"] == task].copy()
    df_task["duration_min"] = df_task["duration_sec"] / 60

    for row in range(4):
        ax = axes[row, col]
        is_split = row >= 2
        y_val = "duration_min" if row in [0, 2] else "snr_mean"

        pt.RainCloud(
            x="diagnosis", y=y_val,
            hue="split" if is_split else None,
            data=df_task, order=diag_order_plot,
            palette=split_palette if is_split else diag_colors,
            bw=0.2, width_viol=0.6, orient="v", alpha=0.6, ax=ax
        )

        annotator = Annotator(
            ax, split_pairs_plot if is_split else diag_pairs_plot,
            data=df_task, x="diagnosis", y=y_val,
            hue="split" if is_split else None,
            order=diag_order_plot,
            hue_order=["TRAIN", "TEST"] if is_split else None
        )
        annotator.configure(test='Kruskal', comparisons_correction="bonferroni",
                            text_format='star', loc='outside',
                            pvalue_thresholds=[(0.001, "***"), (0.01, "**"), (1.0, "ns")])
        annotator.apply_and_annotate()

        # Axis labels
        if row == 3:
            ax.set_xlabel("Diagnosis", fontweight='bold')
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("Duration (min)" if y_val == "duration_min" else "SNR (dB)", fontweight='bold')
        else:
            ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
        if ax.get_legend():
            ax.get_legend().remove()
        if row == 0:
            ax.set_title(task, pad=55, fontweight='bold', fontsize=16)

# Row titles
fig.text(0.5, 0.98, 'A. Speech Duration Analysis by Diagnosis', ha='center', fontsize=18, fontweight='heavy')
fig.text(0.5, 0.71, 'B. Audio SNR Analysis by Diagnosis', ha='center', fontsize=18, fontweight='heavy')
fig.text(0.5, 0.43, 'C. Speech Duration Balance (Train vs. Test)', ha='center', fontsize=18, fontweight='heavy')
fig.text(0.5, 0.23, 'D. Audio SNR Balance (Train vs. Test)', ha='center', fontsize=18, fontweight='heavy')

# Legends
diag_handles = [mpatches.Patch(color=diag_colors[i], label=diag_order_plot[i], alpha=0.8) for i in range(3)]
axes[1, 1].legend(handles=diag_handles, loc='upper center', bbox_to_anchor=(0.5, -0.20),
                  ncol=3, frameon=True, edgecolor='black', prop={'weight': 'bold'})
split_handles = [mpatches.Patch(color=split_palette[k], label=k, alpha=0.8) for k in ["TRAIN", "TEST"]]
axes[3, 1].legend(handles=split_handles, loc='upper center', bbox_to_anchor=(0.5, -0.45),
                  ncol=2, frameon=True, edgecolor='black', prop={'weight': 'bold'})

# Adjust layout (manual nudge)
plt.tight_layout(rect=[0, 0.085, 1, 0.96])
row_nudges = {0: -0.02, 1: -0.07, 2: -0.11, 3: -0.12}
for row_idx, nudge in row_nudges.items():
    for ax in axes[row_idx, :]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + nudge, pos.width, pos.height * 1.8])

plt.savefig(results_dir + "comprehensive_audio_stats_final_dark.png", dpi=600, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# 8. Separate Audio Plots (individual rows A–D)
# -----------------------------------------------------------------------------

# Row A: Duration by diagnosis
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for col, task in enumerate(task_order_plot):
    ax = axes[col]
    df_task = audio_df[audio_df["task"] == task].copy()
    df_task["duration_min"] = df_task["duration_sec"] / 60
    pt.RainCloud(x="diagnosis", y="duration_min", data=df_task, order=diag_order_plot,
                 palette=diag_colors, bw=0.2, width_viol=0.6, orient="v", alpha=0.6, ax=ax)
    annotator = Annotator(ax, diag_pairs_plot, data=df_task, x="diagnosis", y="duration_min", order=diag_order_plot)
    annotator.configure(test='Kruskal', comparisons_correction="bonferroni", text_format='star', loc='outside')
    annotator.apply_and_annotate()
    y_max = df_task["duration_min"].max()
    ax.set_ylim(top=y_max * 1.25)
    ax.set_title(task, fontweight='bold', pad=55)
    ax.set_xlabel("Diagnosis", fontweight='bold')
    if col == 0:
        ax.set_ylabel("Duration (min)", fontweight='bold')
    if ax.get_legend():
        ax.get_legend().remove()
fig.suptitle("A. Speech Duration Analysis by Diagnosis", fontsize=16, fontweight='heavy')
plt.tight_layout(rect=[0, 0.1, 1, 0.92])
plt.savefig(results_dir + "row_A.png", dpi=600, bbox_inches='tight')
plt.close()

# Row B: SNR by diagnosis
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for col, task in enumerate(task_order_plot):
    ax = axes[col]
    df_task = audio_df[audio_df["task"] == task].copy()
    pt.RainCloud(x="diagnosis", y="snr_mean", data=df_task, order=diag_order_plot,
                 palette=diag_colors, bw=0.2, width_viol=0.6, orient="v", alpha=0.6, ax=ax)
    annotator = Annotator(ax, diag_pairs_plot, data=df_task, x="diagnosis", y="snr_mean", order=diag_order_plot)
    annotator.configure(test='Kruskal', comparisons_correction="bonferroni", text_format='star', loc='outside')
    annotator.apply_and_annotate()
    y_max = df_task["snr_mean"].max()
    ax.set_ylim(top=max(0, y_max + 1))
    ax.set_xlabel("Diagnosis", fontweight='bold')
    if col == 0:
        ax.set_ylabel("SNR (dB)", fontweight='bold')
    if ax.get_legend():
        ax.get_legend().remove()
handles = [mpatches.Patch(color=diag_colors[i], label=diag_order_plot[i], alpha=0.8) for i in range(3)]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=True)
fig.suptitle("B. Audio SNR Analysis by Diagnosis", fontsize=16, fontweight='heavy')
plt.tight_layout(rect=[0, 0.12, 1, 0.92])
plt.savefig(results_dir + "row_B.png", dpi=600, bbox_inches='tight')
plt.close()

# Row C: Duration by split
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for col, task in enumerate(task_order_plot):
    ax = axes[col]
    df_task = audio_df[audio_df["task"] == task].copy()
    df_task["duration_min"] = df_task["duration_sec"] / 60
    pt.RainCloud(x="diagnosis", y="duration_min", hue="split", data=df_task, order=diag_order_plot,
                 palette=split_palette, bw=0.2, width_viol=0.6, orient="v", alpha=0.6, ax=ax)
    annotator = Annotator(ax, split_pairs_plot, data=df_task, x="diagnosis", y="duration_min",
                          hue="split", order=diag_order_plot, hue_order=["TRAIN", "TEST"])
    annotator.configure(test='Kruskal', comparisons_correction="bonferroni", text_format='star', loc='outside')
    annotator.apply_and_annotate()
    y_max = df_task["duration_min"].max()
    ax.set_ylim(top=y_max * 1.25)
    ax.set_xlabel("Diagnosis", fontweight='bold')
    if col == 0:
        ax.set_ylabel("Duration (min)", fontweight='bold')
    if ax.get_legend():
        ax.get_legend().remove()
fig.suptitle("C. Speech Duration Balance (Train vs. Test)", fontsize=16, fontweight='heavy')
plt.tight_layout(rect=[0, 0.1, 1, 0.92])
plt.savefig(results_dir + "row_C.png", dpi=600, bbox_inches='tight')
plt.close()

# Row D: SNR by split
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for col, task in enumerate(task_order_plot):
    ax = axes[col]
    df_task = audio_df[audio_df["task"] == task].copy()
    pt.RainCloud(x="diagnosis", y="snr_mean", hue="split", data=df_task, order=diag_order_plot,
                 palette=split_palette, bw=0.2, width_viol=0.6, orient="v", alpha=0.6, ax=ax)
    annotator = Annotator(ax, split_pairs_plot, data=df_task, x="diagnosis", y="snr_mean",
                          hue="split", order=diag_order_plot, hue_order=["TRAIN", "TEST"])
    annotator.configure(test='Kruskal', comparisons_correction="bonferroni", text_format='star', loc='outside')
    annotator.apply_and_annotate()
    y_max = df_task["snr_mean"].max()
    ax.set_ylim(top=max(0, y_max + 1))
    ax.set_xlabel("Diagnosis", fontweight='bold')
    if col == 0:
        ax.set_ylabel("SNR (dB)", fontweight='bold')
    if ax.get_legend():
        ax.get_legend().remove()
handles_split = [mpatches.Patch(color=split_palette[k], label=k, alpha=0.8) for k in ["TRAIN", "TEST"]]
fig.legend(handles=handles_split, loc='lower center', ncol=2, frameon=True)
fig.suptitle("D. Audio SNR Balance (Train vs. Test)", fontsize=16, fontweight='heavy')
plt.tight_layout(rect=[0, 0.12, 1, 0.92])
plt.savefig(results_dir + "row_D.png", dpi=600, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 9. Statistical Tests for Audio Duration and SNR
# -----------------------------------------------------------------------------

print("\n--- Audio Statistics ---")

tasks = audio_df['task'].unique()
results_dur = {}
results_snr = {}

for t in tasks:
    df_t = audio_df[audio_df['task'] == t]
    stat_dur, p_dur = kruskal(df_t[df_t.diagnosis=="HC"].duration_sec,
                              df_t[df_t.diagnosis=="MCI"].duration_sec,
                              df_t[df_t.diagnosis=="Dementia"].duration_sec)
    results_dur[t] = (stat_dur, p_dur)
    stat_snr, p_snr = kruskal(df_t[df_t.diagnosis=="HC"].snr_mean,
                              df_t[df_t.diagnosis=="MCI"].snr_mean,
                              df_t[df_t.diagnosis=="Dementia"].snr_mean)
    results_snr[t] = (stat_snr, p_snr)

print("\nKruskal–Wallis results:")
for t in tasks:
    print(f"{t}: Duration H={results_dur[t][0]:.3f}, p={results_dur[t][1]:.3e}; "
          f"SNR H={results_snr[t][0]:.3f}, p={results_snr[t][1]:.3e}")

# Dunn post‑hoc tests
for t in tasks:
    df_t = audio_df[audio_df['task'] == t]
    dunn_dur = sp.posthoc_dunn(df_t, val_col='duration_sec', group_col='diagnosis', p_adjust='bonferroni')
    dunn_snr = sp.posthoc_dunn(df_t, val_col='snr_mean', group_col='diagnosis', p_adjust='bonferroni')
    print(f"\n=== {t} ===")
    print("Duration Dunn:\n", dunn_dur)
    print("SNR Dunn:\n", dunn_snr)

print("\nAll analyses completed.")