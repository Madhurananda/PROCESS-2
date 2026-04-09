#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_gen_audio_info.py

Extract audio information from the PROCESS‑2 dataset:
- Duration per audio file (SFT, PFT, CTD tasks)
- SNR (Signal‑to‑Noise Ratio) using Silero VAD and torchmetrics
- Merge results into final metadata with audio quality flags
- Plot a sample spectrogram and compute file size statistics

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import time
from datetime import datetime
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

# VAD and SNR
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from torch import tensor
from torchmetrics.audio import SignalNoiseRatio

# -----------------------------------------------------------------------------
# 1. Utility Functions (self‑contained)
# -----------------------------------------------------------------------------

def normalise_sig(y):
    """Normalise a signal to a maximum absolute value of 0.99."""
    mult = 0.99 / (max(abs(y)))
    return np.array([x * mult for x in y], dtype=np.float32)


def make_2_list_same_len(a, b):
    """Extend the shorter list with its mean to match the longer list's length."""
    if len(a) > len(b):
        b.extend([np.mean(b)] * (len(a) - len(b)))
    else:
        a.extend([np.mean(a)] * (len(b) - len(a)))
    return a, b


def do_calc_Audio_analysis(args):
    """
    Process one participant directory:
    - Load all .wav files
    - Run VAD to get speech/pause timestamps
    - Compute SNR between speech and pause segments
    Returns a DataFrame with one row per audio file (or per VAD segment).
    """
    a_dir = args[0]
    audio_files = glob.glob(rsmpld_dir + a_dir + '/*.wav')
    df_temp = pd.DataFrame([], columns=OUT__audio_COL_NAMES)

    for audio_file in audio_files:
        wav = read_audio(audio_file)
        speech_ts = get_speech_timestamps(wav, VAD_model, return_seconds=False)

        Y, SR = librosa.load(audio_file, sr=VAD_sr)
        Y = normalise_sig(Y)

        starts, ends = [], []
        speech_seg = []
        pause_seg = []
        data = []

        if len(speech_ts) > 0:
            for i, seg in enumerate(speech_ts):
                start_t = seg['start'] / VAD_sr
                end_t   = seg['end']   / VAD_sr
                starts.append(start_t)
                ends.append(end_t)

                speech_seg.extend(Y[seg['start']:seg['end']])

                if i == 0:
                    pause_seg.extend(Y[:seg['start']])
                else:
                    pause_seg.extend(Y[speech_ts[i-1]['end']:seg['start']])

                if i == len(speech_ts) - 1:
                    pause_seg.extend(Y[seg['end']:])

            # Build data matrix for all segments (identical values repeated per segment)
            n_seg = len(starts)
            data.append([a_dir] * n_seg)
            # Extract question ID (last part of filename before .wav)
            q_id = audio_file.split('/')[-1].split('.wav')[0].split('_')[-1]
            data.append([q_id] * n_seg)
            data.append(['VAD_OUT'] * n_seg)
            data.append(starts)
            data.append(ends)
            data.append([round(len(speech_seg)/VAD_sr, 2)] * n_seg)
            data.append([round(len(pause_seg)/VAD_sr, 2)] * n_seg)
            data.append([round(len(Y)/VAD_sr, 2)] * n_seg)

            # Compute SNR (target = pause, preds = speech)
            target, preds = make_2_list_same_len(pause_seg, speech_seg)
            snr_val = SignalNoiseRatio()(tensor(preds), tensor(target)).tolist()
            data.append([round(snr_val, 2)] * n_seg)

        else:
            # No speech detected
            q_id = audio_file.split('/')[-1].split('.wav')[0].split('_')[-1]
            data.append([a_dir])
            data.append([q_id])
            data.append(['NO_VAD'])
            data.append([0])
            data.append([0])
            data.append([0])
            data.append([0])
            data.append([0])
            data.append([0])

        # Convert to DataFrame and append
        df_chunk = pd.DataFrame(np.array(data).transpose(), columns=OUT__audio_COL_NAMES)
        df_temp = pd.concat([df_temp, df_chunk], ignore_index=True, sort=False)

    return df_temp


def do_multi_Audio_analysis():
    """Run audio analysis in parallel over all participants."""
    df_audio = pd.DataFrame([], columns=OUT__audio_COL_NAMES)
    inputs = zip(list(df_metadata.dir_name))
    n_jobs = min(N_jobs, len(df_metadata), cpu_count())
    print(f'n_jobs: {n_jobs}')
    sys.stdout.flush()

    results = tqdm(Pool(n_jobs).imap_unordered(do_calc_Audio_analysis, inputs),
                   total=len(df_metadata))
    for res in results:
        if len(res) > 0:
            df_audio = pd.concat([df_audio, res], ignore_index=True, sort=False)

    df_audio.to_csv(temp_df_path, index=False)


def plot_spectrogram(audio_path, save_path=None, duration=None, log_freq=False,
                     cmap='viridis', dpi=600, figsize=(12, 6)):
    """
    Plot a spectrogram of an audio file.
    - duration: load only first N seconds (None = full file)
    - log_freq: use mel scale (True) or linear Hz (False)
    """
    y, sr = librosa.load(audio_path, sr=None, duration=duration)
    D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048, window='hann')
    S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    if log_freq:
        img = librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time',
                                       y_axis='mel', ax=ax, cmap=cmap)
        ylabel = 'Mel Frequency (kHz)'
    else:
        img = librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time',
                                       y_axis='hz', ax=ax, cmap=cmap)
        ylabel = 'Frequency (kHz)'

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:g}'))
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(os.path.basename(audio_path), fontsize=14, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    else:
        plt.show()


def get_file_size_mb(file_path):
    """Return file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def compute_stats(sizes_mb):
    """Return (min, max, avg) from a list of sizes in MB."""
    if not sizes_mb:
        return (None, None, None)
    return (min(sizes_mb), max(sizes_mb), sum(sizes_mb) / len(sizes_mb))


def do_get_file_sizes(parent_folder):
    """Print statistics of .wav file sizes per task (SFT, PFT, CTD)."""
    task_sizes = defaultdict(list)
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith('.wav'):
                base = os.path.splitext(file)[0]
                parts = base.split('__')
                if len(parts) >= 3:
                    task = parts[-1]
                    if task in ['SFT', 'PFT', 'CTD']:
                        full_path = os.path.join(root, file)
                        task_sizes[task].append(get_file_size_mb(full_path))

    print("\n=== Audio File Size Statistics (in MB) ===\n")
    print(f"{'Task':<6} {'Count':<8} {'Min (MB)':<12} {'Max (MB)':<12} {'Avg (MB)':<12}")
    print("-" * 50)
    for task in ['SFT', 'PFT', 'CTD']:
        sizes = task_sizes.get(task, [])
        count = len(sizes)
        if count == 0:
            print(f"{task:<6} {'No files found':<40}")
        else:
            mn, mx, avg = compute_stats(sizes)
            print(f"{task:<6} {count:<8} {mn:<12.2f} {mx:<12.2f} {avg:<12.2f}")


# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__' and '__file__' in globals():
    # --- Command line argument ---
    if len(sys.argv) < 2:
        print("Usage: python PROCESS2_gen_audio_info.py N_JOBS")
        sys.exit(1)
    N_jobs = int(sys.argv[1])

    # --- Paths and constants ---
    data_dir = '../data/'
    # TODO: Replace with actual paths if different
    rsmpld_dir = "../path/to/resampled/audio/"          # directory with resampled audio (16 kHz)
    temp_df_path = data_dir + 'TEMP_audio_INFO.csv'    # intermediate CSV for SNR results

    VAD_sr = 16000
    OUT__audio_COL_NAMES = [
        'dir_name', 'Question', 'TEXT', 'START (sec)', 'END (sec)',
        'Speech_Audio', 'Pause_Audio', 'Total_Audio', 'SNR'
    ]

    # --- Load metadata ---
    df_metadata = pd.read_csv(data_dir + 'meta-info.csv').rename(columns={'IDs': 'dir_name'})

    # -------------------------------------------------------------------------
    # 2.1 Extract basic audio duration per task
    # -------------------------------------------------------------------------
    print("Extracting audio duration per task ...")
    records = []
    for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata)):
        d = row['dir_name']
        participant_dir = os.path.join(data_dir, d)
        for f in os.listdir(participant_dir):
            if f.endswith(".wav"):
                path = os.path.join(participant_dir, f)
                y, sr = librosa.load(path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)

                if "SFT" in f:
                    task = "SFT"
                elif "PFT" in f:
                    task = "PFT"
                elif "CTD" in f:
                    task = "CTD"
                else:
                    task = "UNKNOWN"

                records.append({
                    "participant": d,
                    "task": task,
                    "diagnosis": row['diagnosis'],
                    "split": row['Split'],
                    "duration_sec": duration
                })

    audio_df = pd.DataFrame(records)
    audio_df.to_csv(data_dir + 'Audio_duration.csv', index=False)
    print("Audio duration extraction done.\n")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # 2.2 Calculate SNR using VAD (requires Silero VAD model)
    # -------------------------------------------------------------------------
    print("Calculating SNR (this may take a while) ...")
    df_metadata = df_metadata.sort_values(by="dir_name").reset_index(drop=True)
    VAD_model = load_silero_vad()
    do_multi_Audio_analysis()
    print("SNR extraction done.\n")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # 2.3 Merge SNR info with metadata, flag problematic recordings
    # -------------------------------------------------------------------------
    df_audio = pd.read_csv(temp_df_path)
    df_audio = df_audio.sort_values(by=['SNR', 'dir_name'], ascending=False)
    df_audio = df_audio[['dir_name', 'Question', 'TEXT', 'SNR']]
    df_audio.drop_duplicates(inplace=True)
    df_audio = df_audio.merge(df_metadata[['dir_name', 'diagnosis']], on='dir_name', how='inner')
    df_audio['SNR'] = df_audio['SNR'].astype(float)

    # Participants with no VAD or SNR == 0
    df_bad = df_audio[(df_audio['TEXT'] == 'NO_VAD') | (df_audio['SNR'] == 0)]
    df_bad['AUDIO_SNR'] = 'ISSUES'
    df_bad = df_bad[['dir_name', 'AUDIO_SNR']].drop_duplicates()

    # Remaining participants: compute mean ± std of SNR across questions
    df_good = df_audio[~df_audio.dir_name.isin(df_bad.dir_name)][['dir_name', 'Question', 'SNR']]
    stats = df_good.groupby('dir_name')['SNR'].agg(['mean', 'std'])
    df_good = df_good.join(stats, on='dir_name')
    df_good['AUDIO_SNR'] = df_good.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)
    df_good = df_good[['dir_name', 'AUDIO_SNR']].drop_duplicates()

    # Combine and save final metadata
    df_audio_quality = pd.concat([df_bad, df_good], ignore_index=True).sort_values('dir_name')
    df_final_meta = df_metadata.merge(df_audio_quality, on='dir_name', how='inner')
    df_final_meta.to_csv(data_dir + 'meta-info_FINAL.csv', index=False)
    print("Final metadata with audio quality saved.\n")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # 2.4 Optional: Plot spectrogram for a sample file
    # -------------------------------------------------------------------------
    sample_audio = data_dir + 'PROCESS-2/PROCESS-2_rec__002/PROCESS-2_rec__002__CTD.wav'
    output_fig = data_dir + 'FIGS/spectrogram.png'
    if os.path.isfile(sample_audio):
        plot_spectrogram(sample_audio, save_path=output_fig, duration=10.0,
                         log_freq=False, cmap='viridis')
    else:
        print(f"Sample audio not found: {sample_audio} – skipping spectrogram.")

    # -------------------------------------------------------------------------
    # 2.5 File size statistics for all .wav files in PROCESS-2 folder
    # -------------------------------------------------------------------------
    process2_dir = "../data/PROCESS-2"
    if os.path.isdir(process2_dir):
        do_get_file_sizes(process2_dir)
    else:
        print(f"PROCESS-2 folder not found: {process2_dir} – skipping size stats.")

    print("\nAll tasks completed.")