#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_BASELINE_class.py

Baseline classification and regression pipeline for the PROCESS‑2 dataset.
Supports:
- Acoustic features (openSMILE: ComParE_2016, eGeMAPSv02)
- Text features (BoW / TF‑IDF) with multiple ASR transcriptions (MAN, Whisper, Wav2Vec2)
- Classification: 2‑way (HC vs. MCI/Dementia) and 3‑way (HC / MCI / Dementia)
- Regression: MMSE score prediction (Random Forest, Decision Tree)
- Multi‑processing for feature extraction

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import time
import glob
import math
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import librosa
import opensmile
from tqdm import tqdm

# Scikit‑learn
from sklearn.preprocessing import robust_scale
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)

# -----------------------------------------------------------------------------
# 1. Utility Functions
# -----------------------------------------------------------------------------

def diff_list(list_1, list_2):
    """Return elements of list_1 that are not in list_2."""
    return [x for x in list(list_1) if x not in list(list_2)]


def read_file(file):
    """Read a text file and return its content as a string."""
    with open(file, "r") as f:
        return f.read()


def find_values_counts(given_list, mult_only=0):
    """
    Count occurrences of unique values in a list.
    If mult_only == 1, only return values that appear more than once.
    """
    freq_dist = []
    counted_list = np.unique(given_list, return_counts=True)
    for i in range(len(counted_list[0])):
        if mult_only == 1:
            if counted_list[1][i] > 1:
                freq_dist.append([counted_list[0][i], counted_list[1][i]])
        else:
            freq_dist.append([counted_list[0][i], counted_list[1][i]])
    return freq_dist


def calc_metrics(actual_labels, pred_vals, avg=None):
    """
    Calculate classification metrics:
    - F1, Precision, Recall (macro or micro)
    - Confusion matrix, specificity, sensitivity, accuracy
    """
    if avg is None:
        f1 = f1_score(actual_labels, pred_vals)
        prec = precision_score(actual_labels, pred_vals)
        rec = recall_score(actual_labels, pred_vals)
    else:
        f1 = f1_score(actual_labels, pred_vals, average=avg)
        prec = precision_score(actual_labels, pred_vals, average=avg)
        rec = recall_score(actual_labels, pred_vals, average=avg)

    conf = confusion_matrix(actual_labels, pred_vals)
    # Specificity and sensitivity only defined for binary classification
    if conf.shape == (2, 2):
        specificity = conf[0][0] / (conf[0][0] + conf[0][1])
        sensitivity = conf[1][1] / (conf[1][0] + conf[1][1])
    else:
        specificity = sensitivity = 0.0

    acc = accuracy_score(actual_labels, pred_vals)
    return f1, prec, rec, acc, conf, specificity, sensitivity


def majority_voting_pred_labels(df, a_class_type, verbose):
    """
    Aggregate predictions at the participant level (majority voting) and compute
    final classification metrics. Assumes df contains columns:
    'r_IDs', 'pred_label', 'pred_proba', 'labels'
    """
    # Average predictions and probabilities per participant
    grouped = df.groupby('r_IDs').agg({
        'pred_label': 'mean',
        'pred_proba': 'mean'
    }).reset_index()
    grouped.columns = ['r_IDs', 'mean_pred_label', 'mean_pred_proba']

    df_final = df.merge(grouped, on='r_IDs', how='inner').drop_duplicates('r_IDs')

    # Determine final label based on class type and threshold
    if df.shape[0] > df_final.shape[0]:          # multiple questions per participant
        if a_class_type == '3-way':
            thr = 1/3
            final_labels = []
            for x in df_final.mean_pred_label:
                if x < thr:
                    final_labels.append(0)
                elif x < 2*thr:
                    final_labels.append(1)
                else:
                    final_labels.append(2)
        else:                                    # 2-way
            thr = 0.5
            final_labels = [0 if x < thr else 1 for x in df_final.mean_pred_label]
        df_final['final_pred_label'] = final_labels
    else:
        df_final['final_pred_label'] = df['pred_label'].values

    # Calculate metrics
    f1, prec, rec, acc, conf, spec, sens = calc_metrics(
        list(df_final.labels), list(df_final.final_pred_label), avg='macro'
    )

    # AUC handling
    if a_class_type == '2-way':
        auc = roc_auc_score(list(df_final.labels), list(df_final.mean_pred_proba))
    else:
        auc = 0   # 3‑way AUC not implemented here
        if verbose == 1:
            from sklearn.metrics import precision_recall_fscore_support as score
            precision, recall, fscore, _ = score(
                list(df_final.labels), list(df_final.final_pred_label)
            )
            print('Metric \t HC \t MCI \t Demen')
            print(f'Precis \t {precision[0]:.2f} \t {precision[1]:.2f} \t {precision[2]:.2f}')
            print(f'Recall \t {recall[0]:.2f} \t {recall[1]:.2f} \t {recall[2]:.2f}')
            print(f'F1-val \t {fscore[0]:.2f} \t {fscore[1]:.2f} \t {fscore[2]:.2f}')

    return f1, prec, rec, acc, conf, auc, spec, sens


def calc_regress_metrics(y_true, y_pred):
    """Compute regression metrics: MSE, RMSE, MAE, R²."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


# -----------------------------------------------------------------------------
# 2. Label Preparation (2‑way / 3‑way)
# -----------------------------------------------------------------------------

DEM_CAT = ['Dementia', 'Mild_Vas_Dementia', 'Dementia (FTD)', 'Park_Dementia']
UNKN_CAT = ['FMD', 'FCD', 'Unknown', 'TBC', 'NC_FMD']


def prep_labels_2_WAY(df_metadata):
    """
    Convert diagnosis to binary labels:
    0 = HC, 1 = MCI or any dementia, -1 = unknown/unwanted.
    """
    if 'labels' in df_metadata.columns:
        df_metadata = df_metadata.drop(['labels'], axis=1)

    df = df_metadata.reset_index(drop=True)
    df['labels'] = df['diagnosis']
    df.loc[df.labels == 'HC', 'labels'] = 0
    df.loc[df.labels == 'MCI', 'labels'] = 1
    for cat in DEM_CAT:
        df.loc[df.labels == cat, 'labels'] = 1
    for cat in UNKN_CAT:
        df.loc[df.labels == cat, 'labels'] = -1

    # Sanity check
    if not all(isinstance(x, int) for x in set(df.labels)):
        print('Not all diagnoses have been mapped to labels. Check:')
        print(df.labels.value_counts())
        sys.stdout.flush()
        time.sleep(100000000)
    return df


def prep_labels_3_WAY(df_metadata):
    """
    Convert diagnosis to three‑class labels:
    0 = HC, 1 = MCI, 2 = any dementia, -1 = unknown.
    """
    if 'labels' in df_metadata.columns:
        df_metadata = df_metadata.drop(['labels'], axis=1)

    df = df_metadata.reset_index(drop=True)
    df['labels'] = df['diagnosis']
    df.loc[df.labels == 'HC', 'labels'] = 0
    df.loc[df.labels == 'MCI', 'labels'] = 1
    for cat in DEM_CAT:
        df.loc[df.labels == cat, 'labels'] = 2
    for cat in UNKN_CAT:
        df.loc[df.labels == cat, 'labels'] = -1

    if not all(isinstance(x, int) for x in set(df.labels)):
        print('Not all diagnoses have been mapped to labels. Check:')
        print(df.labels.value_counts())
        sys.stdout.flush()
        time.sleep(100000000)
    return df


def prep_CALC_labels_2_WAY(df_metadata, calc_diag, calc_lbl):
    """Same as prep_labels_2_WAY but with column name arguments (legacy)."""
    if calc_lbl in df_metadata.columns:
        df_metadata = df_metadata.drop([calc_lbl], axis=1)
    df = df_metadata.reset_index(drop=True)
    df[calc_lbl] = df[calc_diag]
    df.loc[df[calc_lbl] == 'HC', calc_lbl] = 0
    df.loc[df[calc_lbl] == 'MCI', calc_lbl] = 1
    for cat in DEM_CAT:
        df.loc[df[calc_lbl] == cat, calc_lbl] = 1
    for cat in UNKN_CAT:
        df.loc[df[calc_lbl] == cat, calc_lbl] = -1
    return df


def prep_CALC_labels_3_WAY(df_metadata, calc_diag, calc_lbl):
    """Same as prep_labels_3_WAY but with column name arguments (legacy)."""
    if calc_lbl in df_metadata.columns:
        df_metadata = df_metadata.drop([calc_lbl], axis=1)
    df = df_metadata.reset_index(drop=True)
    df[calc_lbl] = df[calc_diag]
    df.loc[df[calc_lbl] == 'HC', calc_lbl] = 0
    df.loc[df[calc_lbl] == 'MCI', calc_lbl] = 1
    for cat in DEM_CAT:
        df.loc[df[calc_lbl] == cat, calc_lbl] = 2
    for cat in UNKN_CAT:
        df.loc[df[calc_lbl] == cat, calc_lbl] = -1
    return df


# -----------------------------------------------------------------------------
# 3. Feature Extraction Helpers
# -----------------------------------------------------------------------------

def get_feature_name(df_feat, openSmile_feat):
    """
    Return list of feature column names for a given feature type.
    Supports openSMILE sets, BoW, TF‑IDF, and combined TEXT_AUDIO.
    """
    if openSmile_feat in ('ComParE_2016', 'eGeMAPSv02'):
        if openSmile_feat == 'ComParE_2016':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals
            )
        else:
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
        return list(smile.feature_names)
    elif openSmile_feat == 'NLP-BoW':
        return [x for x in df_feat.columns if x.startswith('BoW_f__')]
    elif openSmile_feat == 'TF-IDF':
        return [x for x in df_feat.columns if x.startswith('TFIDF_f__')]
    elif openSmile_feat == 'TEXT_AUDIO':
        return diff_list(df_feat.columns, ['dir_name', 'Q_type'])
    else:
        raise ValueError(f"Unknown feature type: {openSmile_feat}")


def do_extract_opensmile(args):
    """
    Multiprocessing worker for openSMILE feature extraction.
    Extracts features for one participant and one question type.
    """
    idx, smile, fT_avg, l_q_types, df_meta, audio_dir = args
    df_temp = pd.DataFrame(columns=['dir_name', 'Q_type'] + smile.feature_names)

    if 'EACH_FRAME__' in fT_avg:
        df_temp.insert(2, 'bin_no', '')
        frame_len_sec = int(fT_avg.split('__')[-1])

    for q in l_q_types:
        # Skip non‑standard task names (used elsewhere)
        if q in ('Memory_Task', 'Fluency_Task', 'Picture_Description', 'ALL'):
            continue

        # Locate audio file (two possible naming conventions)
        dir_name = df_meta['dir_name'][idx]
        parts = dir_name.split('_')
        if len(parts) == 5:
            base = dir_name.replace(parts[0] + '_', '')
            pattern = f"{audio_dir}{dir_name}/{base}*_{q}*.wav"
        elif len(parts) == 4:
            pattern = f"{audio_dir}{dir_name}/{dir_name}*_{q}*.wav"
        else:
            print(f"Unexpected dir_name format: {dir_name}")
            sys.exit(0)

        audio_file = glob.glob(pattern)[0]
        sig, sr = librosa.load(audio_file, sr=None)

        if fT_avg == 'ENTIRE_AUDIO':
            feat = smile.process_signal(sig, sr).reset_index(drop=True)
            feat.insert(0, 'dir_name', dir_name)
            feat.insert(1, 'Q_type', q)
            df_temp = pd.concat([df_temp, feat], ignore_index=True)
        else:
            # Frame‑wise extraction (e.g., every 5 seconds)
            frame_len = frame_len_sec * sr
            for i in range(math.ceil(len(sig) / frame_len)):
                seg = sig[i*frame_len : (i+1)*frame_len]
                feat = smile.process_signal(seg, sr).reset_index(drop=True)
                feat.insert(0, 'dir_name', dir_name)
                feat.insert(1, 'Q_type', q)
                feat.insert(2, 'bin_no', i)
                df_temp = pd.concat([df_temp, feat], ignore_index=True)

    return df_temp


def generate_features(df_meta, feat_type, fT_avg, n_jobs, out_csv,
                      q_types, audio_dir, text_dir, asr_type=None, feat_path=None):
    """
    Generate or load features for a given feature type.
    - Acoustic: openSMILE (ComParE_2016 / eGeMAPSv02)
    - Text: BoW or TF‑IDF (with ASR variants)
    - Combined: TEXT_AUDIO (acoustic + text)
    """
    def get_file_ext(asr):
        mapping = {'W2V2': '__ASR_W2V2.txt', 'Whisp_Med': '__ASR_Whisper.txt', 'MAN': '.txt'}
        return mapping.get(asr, '.txt')

    # ----- Acoustic features -----
    if feat_type in ('ComParE_2016', 'eGeMAPSv02'):
        if feat_type == 'ComParE_2016':
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals
            )
        else:
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
        feat_names = list(smile.feature_names)

        df_feat = pd.DataFrame(columns=['dir_name', 'Q_type'] + feat_names)
        if 'EACH_FRAME__' in fT_avg:
            df_feat.insert(2, 'bin_no', '')

        df_meta = df_meta.reset_index(drop=True)
        inputs = zip(range(len(df_meta)),
                     [smile]*len(df_meta),
                     [fT_avg]*len(df_meta),
                     [q_types]*len(df_meta),
                     [df_meta]*len(df_meta),
                     [audio_dir]*len(df_meta))

        n_workers = min(n_jobs, len(df_meta), cpu_count())
        print(f'n_jobs: {n_workers}')
        sys.stdout.flush()

        results = tqdm(Pool(n_workers).imap_unordered(do_extract_opensmile, inputs),
                       total=len(df_meta))
        for res in results:
            df_feat = pd.concat([df_feat, res], ignore_index=True)

        df_feat = df_feat.merge(df_meta, on='dir_name', how='inner')
        df_feat.to_csv(out_csv, index=False)
        return df_feat, feat_names

    # ----- Text features (BoW / TF‑IDF) -----
    elif feat_type in ('NLP-BoW', 'TF-IDF'):
        ext = get_file_ext(asr_type) if asr_type else '.txt'
        corpus, ids, q_list = [], [], []

        df_meta = df_meta.reset_index(drop=True)
        for idx in tqdm(df_meta.index):
            for q in q_types:
                if q in ('Memory_Task', 'Fluency_Task', 'Picture_Description', 'ALL'):
                    continue

                dir_name = df_meta['dir_name'][idx]
                parts = dir_name.split('_')
                if len(parts) == 5:
                    base = dir_name.replace(parts[0] + '_', '')
                    pattern = f"{text_dir}{dir_name}/{base}*_{q}*{ext}"
                elif len(parts) == 4:
                    pattern = f"{text_dir}{dir_name}/{dir_name}*_{q}*{ext}"
                else:
                    print(f"Unexpected dir_name: {dir_name}")
                    sys.exit(0)

                try:
                    txt_file = glob.glob(pattern)[0]
                except IndexError:
                    print(f"Warning: no transcript for {pattern}")
                    continue

                text = ''
                for line in read_file(txt_file).split('\n'):
                    if line.strip():
                        # Remove speaker tag and buzzer sound
                        clean = line.split('\t')[-1].replace('Pat:', '').replace('(Buzzer sounds)', '').strip()
                        text += clean

                corpus.append(text)
                ids.append(dir_name)
                q_list.append(q)

        # Vectorisation
        if feat_type == 'NLP-BoW':
            vectorizer = CountVectorizer()
        else:
            vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        feat_names = [f"{'BoW' if feat_type=='NLP-BoW' else 'TFIDF'}_f__{w}"
                      for w in vectorizer.get_feature_names_out()]

        df_feat = pd.DataFrame(X.toarray(), columns=feat_names)
        df_feat.insert(0, 'dir_name', ids)
        df_feat.insert(1, 'Q_type', q_list)
        df_feat.insert(2, 'text_info', corpus)
        df_feat = df_feat.merge(df_meta, on='dir_name', how='inner')
        df_feat.to_csv(out_csv, index=False)
        return df_feat, feat_names

    # ----- Combined acoustic + text features -----
    elif feat_type == 'TEXT_AUDIO':
        if feat_path is None:
            raise ValueError("feat_path required for TEXT_AUDIO")
        # Load precomputed acoustic features
        audio1 = pd.read_csv(feat_path + 'eGeMAPSv02_ENTIRE_AUDIO.csv', low_memory=False)
        audio2 = pd.read_csv(feat_path + 'ComParE_2016_ENTIRE_AUDIO.csv', low_memory=False)
        # Text features for this ASR
        text_csv = f"{feat_path}{feat_type}_{asr_type}_{fT_avg}.csv" if asr_type else f"{feat_path}{feat_type}_{fT_avg}.csv"
        if os.path.exists(text_csv):
            df_text = pd.read_csv(text_csv, low_memory=False)
        else:
            df_text, _ = generate_features(df_meta, 'NLP-BoW', fT_avg, n_jobs, text_csv,
                                           q_types, audio_dir, text_dir, asr_type, feat_path)
        # Merge
        df_feat = audio1.merge(audio2, on=['dir_name', 'Q_type'], how='inner')
        df_feat = df_feat.merge(df_text, on=['dir_name', 'Q_type'], how='inner')
        feat_names = (get_feature_name(audio1, 'eGeMAPSv02') +
                      get_feature_name(audio2, 'ComParE_2016') +
                      get_feature_name(df_text, 'NLP-BoW'))
        df_feat.to_csv(out_csv, index=False)
        return df_feat, feat_names

    else:
        raise ValueError(f"Unsupported feature type: {feat_type}")


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__' and '__file__' in globals():
    # --- Command line arguments ---
    if len(sys.argv) < 2:
        print('\n\nUsage: python baseline_class.py N_JOBS\n\n')
        sys.exit()
    N_JOBS = int(sys.argv[1])

    start_time = datetime.now()
    print(f'\n\nScript starting at: {start_time.strftime("%Y/%m/%d at %H:%M:%S")}\n')

    # --- Paths ---
    DATA_DIR = '../data/PROCESS-2/'
    BASE_DIR = '..'
    FEAT_DIR = f'{BASE_DIR}/data/results/feats/'
    RESULT_DIR = f'{BASE_DIR}/data/results/class_results/'

    # --- Load metadata ---
    df_meta = pd.read_csv(DATA_DIR + 'meta-info.csv').rename(columns={'IDs': 'dir_name'})
    # No need to split into TRAIN/TEST separately here; we use the 'Split' column later.

    # --- Experiment parameters ---
    Q_TYPES = ['SFT', 'PFT', 'CTD']                     # question types
    CLASSIFIER_IDS = [1, 4]                            # 1 = LR, 4 = MLP
    FEATURE_TYPES = ['eGeMAPSv02', 'ComParE_2016', 'NLP-BoW']   # can add 'TEXT_AUDIO'
    FEAT_AVG_TYPES = ['ENTIRE_AUDIO']                  # only whole‑file used
    CLASS_WAYS = ['2-way', '3-way']                    # classification schemes
    RUN_MODES = ['simple', 'regress']                  # classification and regression
    ASR_LIST = ['MAN', 'Whisp_Med', 'W2V2']            # for text features

    # --- Results container ---
    df_results = pd.DataFrame(columns=[
        'ASR', 'classifier', 'Question', 'model_train', 'feature', 'feat_avg_type',
        'class_type', 'avg_feat', 'F1-value', 'Precision', 'Recall',
        'Accuracy', 'Conf_Mat', 'AUC', 'Specificity', 'Sensitivity',
        'MSE', 'RMSE', 'MAE', 'R2'
    ])
    out_csv = RESULT_DIR + 'PROCESS__stan_class_results.csv'

    # --- Main loops over features, averaging, ASR variants ---
    for feat in FEATURE_TYPES:
        for avg_type in FEAT_AVG_TYPES:
            # Determine which ASR variants to try (only for text features)
            text_feats = ['NLP-BoW', 'TF-IDF', 'TEXT_AUDIO']
            asr_iter = ASR_LIST if feat in text_feats else [None]

            for asr in asr_iter:
                # Feature file name
                if asr is not None:
                    feat_csv = f"{FEAT_DIR}{feat}_{asr}_{avg_type}.csv"
                else:
                    feat_csv = f"{FEAT_DIR}{feat}_{avg_type}.csv"

                # Load or generate features
                if os.path.exists(feat_csv):
                    df_feat = pd.read_csv(feat_csv, low_memory=False)
                    feature_cols = get_feature_name(df_feat, feat)
                else:
                    df_feat, feature_cols = generate_features(
                        df_meta, feat, avg_type, N_JOBS, feat_csv,
                        Q_TYPES, DATA_DIR, DATA_DIR, asr, FEAT_DIR
                    )

                # Remove rows with missing feature values
                df_feat = df_feat.dropna(subset=feature_cols, axis=0)

                # Sanity check for entire‑audio features
                if avg_type == 'ENTIRE_AUDIO' and df_feat.shape[0] / 3 != df_meta.shape[0]:
                    print(f"Warning: {feat} (ASR={asr}) has unexpected row count. Skipping.")
                    continue

                # ---------- REGRESSION (MMSE) ----------
                if 'regress' in RUN_MODES:
                    df_reg = df_feat.copy()
                    for reg_id in [1, 4]:   # 1 = RandomForest, 4 = DecisionTree
                        reg_name = {1: 'RandomForest', 4: 'DecisionTree'}[reg_id]
                        for q in Q_TYPES + ['Fluency_Task', 'ALL']:
                            # Train/test split based on question and Split column
                            if q == 'ALL':
                                train = df_reg[df_reg['Split'] == 'TRAIN']
                                test  = df_reg[df_reg['Split'] == 'TEST']
                            elif q == 'Fluency_Task':
                                train = df_reg[(df_reg['Split'] == 'TRAIN') &
                                              (df_reg.Q_type.isin(['SFT', 'PFT']))]
                                test  = df_reg[(df_reg['Split'] == 'TEST') &
                                              (df_reg.Q_type.isin(['SFT', 'PFT']))]
                            else:
                                train = df_reg[(df_reg['Split'] == 'TRAIN') & (df_reg.Q_type == q)]
                                test  = df_reg[(df_reg['Split'] == 'TEST') & (df_reg.Q_type == q)]

                            # Keep only rows with valid MMSE
                            train = train[train['MMSE'].notna()]
                            test  = test[test['MMSE'].notna()]
                            if len(train) == 0 or len(test) == 0:
                                print(f"Skipping regression for {q}: no MMSE values.")
                                continue

                            print(f"\n--- Regression: {feat} | ASR: {asr} | {q} | {reg_name} ---")
                            print(f"Train samples: {len(train)}, Test samples: {len(test)}")

                            X_train = robust_scale(train[feature_cols].values)
                            y_train = train['MMSE'].values
                            X_test  = robust_scale(test[feature_cols].values)
                            y_test  = test['MMSE'].values

                            if reg_id == 1:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            else:
                                model = DecisionTreeRegressor(max_depth=50, random_state=42)

                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            mse, rmse, mae, r2 = calc_regress_metrics(y_test, preds)

                            asr_str = asr if asr is not None else ''
                            df_results.loc[len(df_results)] = [
                                asr_str, reg_name, q, 'regress', feat, avg_type,
                                'regress', False,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                round(mse, 2), round(rmse, 2), round(mae, 2), round(r2, 2)
                            ]
                            df_results.to_csv(out_csv, index=False)

                # ---------- CLASSIFICATION (2‑way / 3‑way) ----------
                if 'simple' in RUN_MODES:
                    for class_way in CLASS_WAYS:
                        for clf_id in CLASSIFIER_IDS:
                            clf_name = {1: 'LR', 2: 'KNN', 3: 'Linear_SVC', 4: 'MLP',
                                        5: 'MLP_TF', 6: 'TF_CNN', 8: 'Bi_LSTM'}.get(clf_id, 'Unknown')
                            for q in Q_TYPES + ['Fluency_Task', 'ALL']:
                                df_cls = df_feat.copy()

                                # Prepare labels
                                if class_way == '2-way':
                                    df_cls = prep_labels_2_WAY(df_cls)
                                else:
                                    df_cls = prep_labels_3_WAY(df_cls)
                                df_cls = df_cls[df_cls.labels != -1]   # remove unknown

                                # Split train/test
                                if q == 'ALL':
                                    train = df_cls[df_cls['Split'] == 'TRAIN']
                                    test  = df_cls[df_cls['Split'] == 'TEST']
                                elif q == 'Fluency_Task':
                                    train = df_cls[(df_cls['Split'] == 'TRAIN') &
                                                  (df_cls.Q_type.isin(['SFT', 'PFT']))]
                                    test  = df_cls[(df_cls['Split'] == 'TEST') &
                                                  (df_cls.Q_type.isin(['SFT', 'PFT']))]
                                else:
                                    train = df_cls[(df_cls['Split'] == 'TRAIN') & (df_cls.Q_type == q)]
                                    test  = df_cls[(df_cls['Split'] == 'TEST') & (df_cls.Q_type == q)]

                                # Rename column for majority voting function
                                train = train.rename(columns={'dir_name': 'r_IDs'})
                                test  = test.rename(columns={'dir_name': 'r_IDs'})

                                print(f"\n--- Classification: {feat} | ASR: {asr} | {q} | {clf_name} | {class_way} ---")
                                print("Train label counts:", find_values_counts(train.labels))
                                print("Test label counts:", find_values_counts(test.labels))

                                X_train = robust_scale(train[feature_cols].values)
                                y_train = train.labels.values
                                X_test  = robust_scale(test[feature_cols].values)

                                # Train model
                                if clf_id == 1:   # Logistic Regression
                                    model = LogisticRegression(max_iter=int(1e20), n_jobs=N_JOBS,
                                                               class_weight='balanced')
                                    model.fit(X_train, y_train)
                                elif clf_id == 4: # MLP
                                    model = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-5,
                                                          random_state=1, max_iter=int(1e20))
                                    model.fit(X_train, y_train)
                                else:
                                    print(f"Classifier {clf_name} not implemented. Skipping.")
                                    continue

                                preds = model.predict(X_test)
                                proba = model.predict_proba(X_test)[:, 1]  # probability of class 1

                                # Prepare dataframe for majority voting
                                df_test_out = test[['r_IDs', 'Q_type', 'labels']].copy()
                                df_test_out['pred_label'] = preds
                                df_test_out['pred_proba'] = proba

                                f1, prec, rec, acc, conf, auc, spec, sens = majority_voting_pred_labels(
                                    df_test_out, class_way, verbose=0
                                )

                                print(f"F1: {f1:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
                                print(f"Accuracy: {acc:.2f}, AUC: {auc:.2f}")
                                print("Confusion matrix:\n", conf)

                                asr_str = asr if asr is not None else ''
                                df_results.loc[len(df_results)] = [
                                    asr_str, clf_name, q, 'simple', feat, avg_type,
                                    class_way, False,
                                    round(f1, 2), round(prec, 2), round(rec, 2),
                                    round(acc, 2), conf, auc,
                                    round(spec, 2), round(sens, 2),
                                    np.nan, np.nan, np.nan, np.nan
                                ]
                                df_results.to_csv(out_csv, index=False)

    # --- Completion ---
    print("\nALL EXPERIMENTS COMPLETED")
    elapsed = datetime.now() - start_time
    print(f"Finished at: {datetime.now().strftime('%Y/%m/%d at %H:%M:%S')}")
    print(f"Execution time: {elapsed}\n")