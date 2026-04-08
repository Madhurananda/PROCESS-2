#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_embed.py

Extract acoustic and linguistic embeddings from the PROCESS‑2 dataset:
- Acoustic: Wav2Vec2 (facebook/wav2vec2-base-960h) – mean pooling over time
- Linguistic: SentenceTransformer (all-MiniLM-L6-v2) – full transcript embedding

Perform dimensionality reduction (PCA + t‑SNE) and generate scatter plots:
- Acoustic embeddings coloured by diagnosis and by train/test split
- Linguistic embeddings (manual transcripts) by diagnosis and split

Finally, compute distances to the HC centroid and run Kruskal‑Wallis + Dunn tests.

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Model Loading and Path Setup
# -----------------------------------------------------------------------------

if __name__ == '__main__' and '__file__' in globals():
    print("Loading models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Acoustic model (Wav2Vec2)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()

    # Linguistic model (SentenceTransformer)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Models loaded successfully!")

    # Paths
    data_dir = '../data/'
    results_dir = data_dir + 'embeddings/'
    os.makedirs(results_dir, exist_ok=True)

    # Metadata
    metadata = pd.read_csv(data_dir + 'meta-info.csv')
    print(f"\nLoaded metadata for {len(metadata)} subjects")

    # Tasks and transcription types
    tasks = ['SFT', 'PFT', 'CTD']
    transcription_suffixes = {
        'manual': '',
        'asr_w2v2': '__ASR_W2V2',
        'asr_whisper': '__ASR_Whisper'
    }

# -----------------------------------------------------------------------------
# 2. Extract Embeddings for All Subjects
# -----------------------------------------------------------------------------

    all_embeddings = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing subjects"):
        subject_id = row['IDs']
        diagnosis = row['diagnosis']
        split = row['Split']

        print(f"\nProcessing {subject_id} ({diagnosis}, {split})")

        for task in tasks:
            # --- Acoustic embedding from audio file ---
            audio_file = f"{data_dir}{subject_id}/{subject_id}__{task}.wav"
            if os.path.exists(audio_file):
                print(f"  Extracting acoustic embeddings from {task} task")
                try:
                    waveform, sr = torchaudio.load(audio_file)
                    # Resample to 16 kHz if needed
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)
                        sr = 16000
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    # Prepare input for Wav2Vec2
                    waveform_np = waveform.squeeze().numpy()
                    if len(waveform_np.shape) == 1:
                        waveform_np = waveform_np.reshape(1, -1)
                    input_values = wav2vec_processor(
                        waveform_np, sampling_rate=16000, return_tensors="pt"
                    ).input_values.to(device)

                    with torch.no_grad():
                        outputs = wav2vec_model(input_values)
                        pooled = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

                    if pooled is not None and len(pooled) > 0:
                        all_embeddings.append({
                            'ID': subject_id,
                            'diagnosis': diagnosis,
                            'split': split,
                            'task': task,
                            'modality': 'acoustic',
                            'transcription_type': 'original_audio',
                            'embedding_vector': pooled,
                            'age': row['age'],
                            'gender': row['gender'],
                            'MMSE': row['MMSE']
                        })
                        print(f"    ✓ Acoustic embedding (dim={len(pooled)})")
                    else:
                        print("    ✗ Failed to extract acoustic embedding")
                except Exception as e:
                    print(f"    Error: {e}")
            else:
                print(f"  Audio file not found: {audio_file}")

            # --- Linguistic embeddings from transcripts ---
            for trans_type, suffix in transcription_suffixes.items():
                transcript_file = f"{data_dir}{subject_id}/{subject_id}__{task}{suffix}.txt"
                if os.path.exists(transcript_file):
                    print(f"  Extracting linguistic embeddings from {task} ({trans_type})")
                    try:
                        with open(transcript_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        emb = sentence_model.encode(text, convert_to_numpy=True)
                        if emb is not None and len(emb) > 0:
                            all_embeddings.append({
                                'ID': subject_id,
                                'diagnosis': diagnosis,
                                'split': split,
                                'task': task,
                                'modality': 'linguistic',
                                'transcription_type': trans_type,
                                'embedding_vector': emb,
                                'age': row['age'],
                                'gender': row['gender'],
                                'MMSE': row['MMSE']
                            })
                            print(f"    ✓ Linguistic embedding (dim={len(emb)})")
                        else:
                            print("    ✗ Failed to extract linguistic embedding")
                    except Exception as e:
                        print(f"    Error: {e}")
                else:
                    print(f"  Transcript not found: {transcript_file}")

        # Save intermediate results every 10 subjects
        if (idx + 1) % 10 == 0:
            pd.DataFrame(all_embeddings).to_pickle(results_dir + 'temp_embeddings.pkl')
            print(f"\n  Saved {len(all_embeddings)} embeddings so far")

# -----------------------------------------------------------------------------
# 3. Save Final Embeddings DataFrame
# -----------------------------------------------------------------------------

    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df.to_pickle(results_dir + 'all_embeddings.pkl')
    embeddings_df.to_csv(results_dir + 'all_embeddings.csv', index=False)

    print(f"\n{'='*80}")
    print(f"Extracted embeddings for {len(all_embeddings)} samples")
    print(f"{'='*80}")
    print("\nDataFrame info:")
    print(embeddings_df.info())
    print("\nUnique combinations:")
    print(f"  Tasks: {embeddings_df['task'].unique()}")
    print(f"  Modalities: {embeddings_df['modality'].unique()}")
    print(f"  Transcription types: {embeddings_df['transcription_type'].unique()}")
    print("\nSample counts per combination:")
    print(embeddings_df.groupby(['task', 'modality', 'transcription_type']).size().reset_index(name='count'))

    sys.stdout.flush()
    time.sleep(100000000)   # Pause to inspect (original behaviour)

# -----------------------------------------------------------------------------
# 4. Dimensionality Reduction (PCA + t‑SNE) for Each Combination
# -----------------------------------------------------------------------------

    # Reload embeddings (original code does this – keep as is)
    embeddings_df = pd.read_pickle(results_dir + 'all_embeddings.pkl')
    print("\nReloaded embeddings for dimensionality reduction.")

    reduced_list = []

    for modality in ['acoustic', 'linguistic']:
        for trans_type in ['original_audio', 'manual', 'asr_w2v2', 'asr_whisper']:
            # Acoustic only uses original_audio
            if modality == 'acoustic' and trans_type != 'original_audio':
                continue

            mask = (embeddings_df['modality'] == modality) & (embeddings_df['transcription_type'] == trans_type)
            df_sub = embeddings_df[mask].copy()
            if len(df_sub) == 0:
                continue

            print(f"\nProcessing {modality} - {trans_type} (n={len(df_sub)})")

            # Stack embeddings
            X = np.vstack(df_sub['embedding_vector'].values)

            # Standardise
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA to 50 components (or less if original dimension is smaller)
            n_comp = min(50, X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

            # t‑SNE on the PCA‑reduced data
            perplex = min(30, len(df_sub) - 1) if len(df_sub) > 1 else 1
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
            X_tsne = tsne.fit_transform(X_pca[:, :min(30, X_pca.shape[1])])

            df_sub['tsne_1'] = X_tsne[:, 0]
            df_sub['tsne_2'] = X_tsne[:, 1]

            # Keep only relevant columns for reduced dataframe
            reduced_list.append(df_sub[['ID', 'diagnosis', 'split', 'task', 'modality',
                                        'transcription_type', 'tsne_1', 'tsne_2',
                                        'age', 'gender', 'MMSE']].copy())

    reduced_df = pd.concat(reduced_list, ignore_index=True)
    reduced_df.to_pickle(results_dir + 'reduced_embeddings.pkl')
    reduced_df.to_csv(results_dir + 'reduced_embeddings.csv', index=False)

    print(f"\n{'='*80}")
    print(f"Saved reduced embeddings for {len(reduced_df)} samples")
    print(f"{'='*80}")
    print(f"Files saved: {results_dir}reduced_embeddings.pkl / .csv")

# -----------------------------------------------------------------------------
# 5. Visualisation: Acoustic and Linguistic t‑SNE Plots
# -----------------------------------------------------------------------------

    # Reload reduced embeddings (again, as in original)
    reduced_df = pd.read_pickle(results_dir + 'reduced_embeddings.pkl')
    print("\nLoaded reduced embeddings for plotting.")

    # ----- Plot 1: Acoustic embeddings by diagnosis -----
    print("\n" + "="*80)
    print("PLOT 1: Acoustic Embeddings - By Diagnosis")
    print("="*80)

    acoustic_df = reduced_df[reduced_df['modality'] == 'acoustic']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors_diag = {'Dementia': '#d32f2f', 'MCI': '#1976d2', 'HC': '#388e3c'}
    task_order = ['SFT', 'PFT', 'CTD']

    for i, task in enumerate(task_order):
        df_task = acoustic_df[acoustic_df['task'] == task]
        for diag, color in colors_diag.items():
            df_diag = df_task[df_task['diagnosis'] == diag]
            axes[i].scatter(df_diag['tsne_1'], df_diag['tsne_2'],
                            c=color, label=diag, alpha=0.8, s=35, edgecolors='none')
        axes[i].set_title(f'{task} Task', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')

    plt.suptitle('Acoustic Embeddings (Wav2Vec2) - by Diagnosis',
                 fontsize=14, fontweight='bold', y=0.93)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(results_dir + 'acoustic_by_diagnosis.png', dpi=600, bbox_inches='tight')
    plt.show()

    # ----- Plot 2: Acoustic embeddings by train/test split -----
    print("\n" + "="*80)
    print("PLOT 2: Acoustic Embeddings - By Train/Test Split")
    print("="*80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors_split = {'TRAIN': '#0097a7', 'TEST': '#f57c00'}

    for i, task in enumerate(task_order):
        df_task = acoustic_df[acoustic_df['task'] == task]
        for split, color in colors_split.items():
            df_split = df_task[df_task['split'] == split]
            axes[i].scatter(df_split['tsne_1'], df_split['tsne_2'],
                            c=color, label=split, alpha=0.8, s=35, edgecolors='none')
        axes[i].set_title(f'{task} Task', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')

    plt.suptitle('Acoustic Embeddings (Wav2Vec2) - by Train/Test Split',
                 fontsize=14, fontweight='bold', y=0.93)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(results_dir + 'acoustic_by_split.png', dpi=600, bbox_inches='tight')
    plt.show()

    # ----- Plot 3: Linguistic embeddings (manual) by diagnosis -----
    print("\n" + "="*80)
    print("PLOT 3: Linguistic Embeddings (Manual) - By Diagnosis")
    print("="*80)

    linguistic_df = reduced_df[(reduced_df['modality'] == 'linguistic') &
                               (reduced_df['transcription_type'] == 'manual')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i, task in enumerate(task_order):
        df_task = linguistic_df[linguistic_df['task'] == task]
        for diag, color in colors_diag.items():
            df_diag = df_task[df_task['diagnosis'] == diag]
            axes[i].scatter(df_diag['tsne_1'], df_diag['tsne_2'],
                            c=color, label=diag, alpha=0.8, s=35, edgecolors='none')
        axes[i].set_title(f'{task} Task', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=12,
               edgecolor='black', prop={'weight': 'bold'})
    plt.suptitle('Linguistic Embeddings (Manual) - by Diagnosis',
                 fontsize=14, fontweight='bold', y=0.93)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(results_dir + 'linguistic_manual_by_diagnosis.png', dpi=600, bbox_inches='tight')
    plt.show()

    # ----- Plot 4: Linguistic embeddings (manual) by split -----
    print("\n" + "="*80)
    print("PLOT 4: Linguistic Embeddings (Manual) - By Train/Test Split")
    print("="*80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i, task in enumerate(task_order):
        df_task = linguistic_df[linguistic_df['task'] == task]
        for split, color in colors_split.items():
            df_split = df_task[df_task['split'] == split]
            axes[i].scatter(df_split['tsne_1'], df_split['tsne_2'],
                            c=color, label=split, alpha=0.8, s=35, edgecolors='none')
        axes[i].set_title(f'{task} Task', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('t-SNE Component 1')
        axes[i].set_ylabel('t-SNE Component 2')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fontsize=12,
               edgecolor='black', prop={'weight': 'bold'})
    plt.suptitle('Linguistic Embeddings (Manual) - by Train/Test Split',
                 fontsize=14, fontweight='bold', y=0.93)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(results_dir + 'linguistic_manual_by_split.png', dpi=600, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------------------------------
# 6. Statistical Tests: Distance to HC Centroid (Kruskal‑Wallis + Dunn)
# -----------------------------------------------------------------------------

    print("\n" + "="*80)
    print("Statistical Tests: Distance to Healthy Control (HC) Centroid")
    print("="*80)

    # Reload full embeddings (again, as in original)
    embeddings_df = pd.read_pickle(results_dir + 'all_embeddings.pkl')

    # Compute 10 principal components for each embedding (for later use? kept as original)
    for modality in ['acoustic', 'linguistic']:
        for trans_type in ['original_audio', 'manual', 'asr_w2v2', 'asr_whisper']:
            if modality == 'acoustic' and trans_type != 'original_audio':
                continue
            mask = (embeddings_df['modality'] == modality) & (embeddings_df['transcription_type'] == trans_type)
            df_sub = embeddings_df.loc[mask].copy()
            if len(df_sub) == 0:
                continue
            X = np.vstack(df_sub['embedding_vector'].values)
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=10)
            X_pca = pca.fit_transform(X_scaled)
            for i in range(10):
                embeddings_df.loc[mask, f'pca_{i+1}'] = X_pca[:, i]

    # Distance to HC centroid and Kruskal‑Wallis test
    tasks = ['SFT', 'PFT', 'CTD']
    trans_types = ['original_audio', 'manual']

    for task in tasks:
        for ttype in trans_types:
            print(f"\n========== {task} | {ttype} ==========")
            df_task = embeddings_df[
                (embeddings_df['task'] == task) &
                (embeddings_df['transcription_type'] == ttype)
            ].copy()
            if len(df_task) == 0:
                continue

            # Embedding matrix
            X = np.vstack(df_task['embedding_vector'].values)
            diagnoses = df_task['diagnosis'].values

            # HC centroid
            hc_centroid = X[diagnoses == 'HC'].mean(axis=0)

            # Euclidean distance to HC centroid
            df_task['dist_to_HC'] = cdist(X, hc_centroid.reshape(1, -1), metric='euclidean').flatten()

            # Kruskal‑Wallis
            groups = [g['dist_to_HC'].values for _, g in df_task.groupby('diagnosis')]
            H, p = stats.kruskal(*groups)
            print(f"\nDistance to HC centroid")
            print(f"H = {H:.3f}, p = {p:.3e}")

            # Dunn post‑hoc if significant
            if p < 0.05:
                dunn = sp.posthoc_dunn(df_task, val_col='dist_to_HC', group_col='diagnosis',
                                       p_adjust='bonferroni')
                print("\nDunn posthoc (Bonferroni):")
                print(dunn)

    print("\nAll tasks completed.")