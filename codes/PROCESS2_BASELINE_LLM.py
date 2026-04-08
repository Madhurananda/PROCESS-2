#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_BASELINE_LLM.py

Classification and regression using foundation models (DistilBERT, RoBERTa) on
transcribed speech from the PROCESS‑2 dataset. Supports multiple ASR systems:
manual, Whisper (medium), and Wav2Vec2.

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import time
import pickle
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    get_linear_schedule_with_warmup,
    RobertaTokenizerFast, RobertaForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    BartTokenizerFast, BartForSequenceClassification
)

# -----------------------------------------------------------------------------
# 1. Utility Functions 
# -----------------------------------------------------------------------------

def find_values_counts(given_list, mult_only=0):
    """Return frequency of unique values in a list."""
    freq = []
    vals, counts = np.unique(given_list, return_counts=True)
    for v, c in zip(vals, counts):
        if mult_only == 1 and c <= 1:
            continue
        freq.append([v, c])
    return freq


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
    if conf.shape == (2, 2):
        specificity = conf[0, 0] / (conf[0, 0] + conf[0, 1])
        sensitivity = conf[1, 1] / (conf[1, 0] + conf[1, 1])
    else:
        specificity = sensitivity = 0.0

    acc = accuracy_score(actual_labels, pred_vals)
    return f1, prec, rec, acc, conf, specificity, sensitivity


def majority_voting_pred_labels(df, a_class_type, verbose):
    """
    Aggregate predictions at participant level (majority voting) and compute
    final classification metrics. Assumes df has columns:
    'r_IDs', 'pred_label', 'pred_proba', 'labels'
    """
    grouped = df.groupby('r_IDs').agg({
        'pred_label': 'mean',
        'pred_proba': 'mean'
    }).reset_index()
    grouped.columns = ['r_IDs', 'mean_pred_label', 'mean_pred_proba']

    df_final = df.merge(grouped, on='r_IDs', how='inner').drop_duplicates('r_IDs')

    # Multiple questions per participant -> apply threshold
    if df.shape[0] > df_final.shape[0]:
        if a_class_type == '3-way':
            thr = 1.0 / 3.0
            final_labels = []
            for x in df_final.mean_pred_label:
                if x < thr:
                    final_labels.append(0)
                elif x < 2 * thr:
                    final_labels.append(1)
                else:
                    final_labels.append(2)
        else:  # 2-way
            thr = 0.5
            final_labels = [0 if x < thr else 1 for x in df_final.mean_pred_label]
        df_final['final_pred_label'] = final_labels
    else:
        df_final['final_pred_label'] = df['pred_label'].values

    # Metrics
    f1, prec, rec, acc, conf, spec, sens = calc_metrics(
        df_final.labels.values, df_final.final_pred_label.values, avg='macro'
    )

    if a_class_type == '2-way':
        auc = roc_auc_score(df_final.labels, df_final.mean_pred_proba)
    else:
        auc = 0.0  # 3‑way AUC not implemented
        if verbose:
            from sklearn.metrics import precision_recall_fscore_support as score
            precision, recall, fscore, _ = score(
                df_final.labels, df_final.final_pred_label
            )
            print('Metric \t HC \t MCI \t Demen')
            print(f'Precis \t {precision[0]:.2f} \t {precision[1]:.2f} \t {precision[2]:.2f}')
            print(f'Recall \t {recall[0]:.2f} \t {recall[1]:.2f} \t {recall[2]:.2f}')
            print(f'F1-val \t {fscore[0]:.2f} \t {fscore[1]:.2f} \t {fscore[2]:.2f}')

    return f1, prec, rec, acc, conf, auc, spec, sens


def read_file(file_path):
    """Read a text file and return its content."""
    with open(file_path, 'r') as f:
        return f.read()


def get_os_cmd(cuda_ids):
    """Return command to set CUDA_VISIBLE_DEVICES environment variable."""
    return f'os.environ["CUDA_VISIBLE_DEVICES"] = "{",".join(map(str, cuda_ids))}"'


# -----------------------------------------------------------------------------
# 2. Label Preparation (2‑way, 3‑way, regression)
# -----------------------------------------------------------------------------

DEM_CAT = ['Dementia', 'Mild_Vas_Dementia', 'Dementia (FTD)', 'Park_Dementia']
UNKN_CAT = ['FMD', 'FCD', 'Unknown', 'TBC', 'NC_FMD']


def prep_labels_2_WAY(df_metadata):
    """Convert diagnosis to binary labels: 0=HC, 1=MCI/Dementia, -1=unknown."""
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
    """Convert diagnosis to three‑class labels: 0=HC, 1=MCI, 2=Dementia, -1=unknown."""
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
    """Legacy version of prep_labels_2_WAY with column name arguments."""
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
    """Legacy version of prep_labels_3_WAY with column name arguments."""
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


def prep_CALC_labels_regress(df_metadata, target_col, calc_lbl):
    """Prepare regression labels (e.g., MMSE). Drops rows with missing target."""
    if calc_lbl in df_metadata.columns:
        df_metadata = df_metadata.drop([calc_lbl], axis=1)
    df = df_metadata.dropna(subset=[target_col]).reset_index(drop=True)
    df[calc_lbl] = df[target_col].astype(float)
    return df


def calc_class_weight(train_y):
    """Compute balanced class weights for imbalanced training data."""
    classes = np.unique(train_y)
    weights = compute_class_weight('balanced', classes=classes, y=train_y)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
    # Fill missing labels with weight 1 (should not happen)
    max_cls = int(np.max(classes))
    for i in range(max_cls):
        if i not in class_weight_dict:
            class_weight_dict[i] = 1.0
    return class_weight_dict


def remove_empty_vals(df):
    """Remove rows where 'text_info' is empty/NaN."""
    empty = df[~df['text_info'].notna()]
    if len(empty) > 0:
        print('CHECK: rows with empty text_info will be excluded:')
        print(empty)
        df = df[df['text_info'].notna()]
    return df


# -----------------------------------------------------------------------------
# 3. PyTorch Datasets and Training Utilities
# -----------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """Dataset for classification (labels as integers)."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_length,
            padding='max_length', truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }


class TextRegressionDataset(Dataset):
    """Dataset for regression (labels as floats)."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_length,
            padding='max_length', truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class EarlyStopping:
    """Early stopping based on a monitored score (higher is better)."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.best_model_state = None

    def step(self, current_score, model):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False   # continue training
        else:
            self.counter += 1
            return self.counter >= self.patience   # stop if patience exhausted


# -----------------------------------------------------------------------------
# 4. Training and Evaluation Functions
# -----------------------------------------------------------------------------

# Global variable used in train() – defined later in main loop
weight_tensor = None

def train(model, data_loader, optimizer, scheduler, device):
    """Train a classification model for one epoch."""
    model.train()
    preds_list = []
    labels_list = []
    losses = []
    probs_list = []

    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Use global weight_tensor for class‑weighted loss
        loss = nn.CrossEntropyLoss(weight=weight_tensor)(outputs.logits, labels)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        scheduler.step()

        _, preds = torch.max(outputs.logits, dim=1)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # probability of class 1

        preds_list.extend(preds.cpu().tolist())
        labels_list.extend(labels.cpu().tolist())
        probs_list.extend(probs.cpu().tolist())

    f1, prec, rec, acc, conf, spec, sens = calc_metrics(labels_list, preds_list, avg='macro')
    return f1, prec, rec, conf, np.mean(losses), np.std(losses), probs_list


def evaluate(model, data_loader, device):
    """Evaluate a classification model."""
    model.eval()
    preds_list = []
    labels_list = []
    losses = []
    probs_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            losses.append(loss.item())

            _, preds = torch.max(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            preds_list.extend(preds.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
            probs_list.extend(probs.cpu().tolist())

    f1, prec, rec, acc, conf, spec, sens = calc_metrics(labels_list, preds_list, avg='macro')
    return f1, prec, rec, conf, np.mean(losses), np.std(losses), preds_list, probs_list


def train_regress(model, data_loader, optimizer, scheduler, device):
    """Train a regression model for one epoch."""
    model.train()
    preds_list = []
    labels_list = []
    losses = []
    criterion = nn.MSELoss()

    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)

        loss = criterion(logits, labels)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        scheduler.step()

        preds_list.extend(logits.cpu().tolist())
        labels_list.extend(labels.cpu().tolist())

    mse = mean_squared_error(labels_list, preds_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels_list, preds_list)
    r2 = r2_score(labels_list, preds_list)
    return mse, rmse, mae, r2, np.mean(losses), np.std(losses), preds_list


def evaluate_regress(model, data_loader, device):
    """Evaluate a regression model."""
    model.eval()
    preds_list = []
    labels_list = []
    losses = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds_list.extend(logits.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    mse = mean_squared_error(labels_list, preds_list)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels_list, preds_list)
    r2 = r2_score(labels_list, preds_list)
    return mse, rmse, mae, r2, np.mean(losses), np.std(losses), preds_list


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__' and '__file__' in globals():
    # --- Command line arguments (GPU IDs) ---
    if len(sys.argv) < 2:
        print('Usage: python PROCESS2_BASELINE_LLM.py "0"   (or "0,1" for multiple GPUs)')
        sys.exit()

    cuda_ids = sys.argv[1].split(',')
    cuda_ids = [int(x) for x in cuda_ids]
    cuda_ids.sort()
    os_cmd = get_os_cmd(cuda_ids)
    try:
        exec(os_cmd)
    except Exception:
        print(f'Failed to set CUDA_VISIBLE_DEVICES with command: {os_cmd}')
        sys.exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    start_time = datetime.now()
    print(f'\nScript starting at: {start_time.strftime("%Y/%m/%d at %H:%M:%S")}\n')

    # --- Paths ---
    BASE_DIR = '..'
    FEAT_DIR = f'{BASE_DIR}/data/Cogno_Challenge_2/results/feats/'
    RESULTS_DIR = f'{BASE_DIR}/data/Cogno_Challenge_2/results/class_results/'
    DATA_DIR = '../data/Cogno_Challenge_2/COGNO_DATA_SHARE_EXPMNT/'

    # --- Hyperparameters ---
    VERBOSE = 0
    MAX_LEN = 512
    N_EPOCHS = 200
    BATCH_SIZE = 64

    # --- Experiment settings ---
    Q_TYPES = ['SFT', 'PFT', 'CTD']
    CLASS_WAYS = ['2-way', '3-way', 'regress']
    MODEL_NAMES = ['DistilBERT', 'RoBERTa']   # BART excluded (optional)
    LEARNING_RATES = [0.0001, 0.00005, 0.00001, 0.000005]
    ASR_TYPES = ['MAN', 'Whisp_Med', 'W2V2']

    # --- Load metadata ---
    df_meta = pd.read_csv(DATA_DIR + 'meta-info.csv').rename(columns={'IDs': 'dir_name'})

    # --- Results DataFrame ---
    df_results = pd.DataFrame(columns=[
        'ASR', 'classifier', 'class_type', 'Learning_rate', 'Question',
        'F1-value', 'Accuracy', 'Precision', 'Recall', 'Conf_Mat', 'AUC',
        'Specificity', 'Sensitivity', 'MSE', 'RMSE', 'MAE', 'R2'
    ])
    results_csv = RESULTS_DIR + 'PROCESS_LLM_seq_results.csv'

    # --- Main loops over ASR, task type, model, LR, and question ---
    for asr in ASR_TYPES:
        # Determine file extension for this ASR
        if asr == 'W2V2':
            ext = '__ASR_W2V2.txt'
        elif asr == 'Whisp_Med':
            ext = '__ASR_Whisper.txt'
        else:   # MAN
            ext = '.txt'

        feat_csv = FEAT_DIR + f'{asr}__PROCESS_LLM_TEXT.csv'

        # Load or generate text features
        if os.path.exists(feat_csv):
            df_feat = pd.read_csv(feat_csv, low_memory=False)
        else:
            df_feat = pd.DataFrame(columns=['dir_name', 'Q_type', 'text_info'])
            df_meta = df_meta.reset_index(drop=True)
            for idx in tqdm(df_meta.index, desc=f'Extracting text for {asr}'):
                for q in Q_TYPES:
                    try:
                        pattern = f"{DATA_DIR}{df_meta['dir_name'][idx]}/{df_meta['dir_name'][idx]}__{q}{ext}"
                        txt_file = glob.glob(pattern)[0]
                        text = ''
                        for line in read_file(txt_file).split('\n'):
                            if line.strip():
                                # Remove speaker tag and buzzer sound
                                clean = line.split('\t')[-1].replace('Pat:', '').replace('(Buzzer sounds)', '').strip()
                                text += clean
                    except Exception:
                        text = ''
                    df_feat.loc[len(df_feat)] = [df_meta['dir_name'][idx], q, text]
            df_feat = df_feat.merge(df_meta, on='dir_name', how='inner')
            df_feat.to_csv(feat_csv, index=False)

        print(f'Feature extraction for ASR={asr} done.')
        sys.stdout.flush()

        # --- Experiment loops ---
        for class_type in CLASS_WAYS:
            for model_name in MODEL_NAMES:
                for lr in LEARNING_RATES:
                    for q_consider in Q_TYPES + ['Fluency_Task', 'ALL']:
                        # Prepare labels according to task type
                        if class_type == '2-way':
                            num_classes = 2
                            try:
                                df_iter = prep_labels_2_WAY(df_feat)
                            except Exception:
                                df_iter = prep_CALC_labels_2_WAY(df_feat, 'diagnosis', 'labels')
                        elif class_type == '3-way':
                            num_classes = 3
                            try:
                                df_iter = prep_labels_3_WAY(df_feat)
                            except Exception:
                                df_iter = prep_CALC_labels_3_WAY(df_feat, 'diagnosis', 'labels')
                        else:   # regress
                            num_classes = 1
                            df_iter = prep_CALC_labels_regress(df_feat, 'MMSE', 'labels')

                        print('\n----------------------')
                        print(f'Classifier : {model_name}')
                        print(f'Learning rate: {lr}')
                        print(f'Question: {q_consider}')

                        # Remove unknown labels
                        if class_type != 'regress':
                            df_iter = df_iter[df_iter.labels != -1]

                        # Split train/test based on Split column and question
                        if q_consider == 'ALL':
                            df_train = df_iter[df_iter['Split'] == 'TRAIN']
                            df_test  = df_iter[df_iter['Split'] == 'TEST']
                        elif q_consider == 'Fluency_Task':
                            df_train = df_iter[(df_iter['Split'] == 'TRAIN') & df_iter.Q_type.isin(['SFT', 'PFT'])]
                            df_test  = df_iter[(df_iter['Split'] == 'TEST')  & df_iter.Q_type.isin(['SFT', 'PFT'])]
                        else:
                            df_train = df_iter[(df_iter['Split'] == 'TRAIN') & (df_iter.Q_type == q_consider)]
                            df_test  = df_iter[(df_iter['Split'] == 'TEST')  & (df_iter.Q_type == q_consider)]

                        df_train = remove_empty_vals(df_train)
                        df_test  = remove_empty_vals(df_test)

                        train_texts = df_train.text_info.tolist()
                        test_texts  = df_test.text_info.tolist()

                        if class_type == 'regress':
                            train_labels = df_train.labels.astype(float).tolist()
                            test_labels  = df_test.labels.astype(float).tolist()
                        else:
                            train_labels = df_train.labels.astype(int).tolist()
                            test_labels  = df_test.labels.astype(int).tolist()
                            print('Train label counts:', find_values_counts(df_train.labels))
                            print('Test label counts: ', find_values_counts(df_test.labels))

                        # Class weights for classification
                        if class_type != 'regress':
                            weights = calc_class_weight(train_labels)
                            weight_list = [weights[i] for i in range(num_classes)]
                            weight_tensor = torch.FloatTensor(weight_list).to(device)

                        # Load tokenizer and model
                        if model_name == 'BART':
                            model_id = "facebook/bart-base"
                            tokenizer = BartTokenizerFast.from_pretrained(model_id)
                            model = BartForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
                        elif model_name == 'RoBERTa':
                            model_id = "roberta-base"
                            tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
                            model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)
                        else:   # DistilBERT
                            model_id = 'distilbert-base-uncased'
                            tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
                            model = DistilBertForSequenceClassification.from_pretrained(model_id, num_labels=num_classes)

                        model = nn.DataParallel(model)
                        model.to(device)

                        # Create datasets and dataloaders
                        if class_type == 'regress':
                            train_dataset = TextRegressionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
                            test_dataset  = TextRegressionDataset(test_texts, test_labels, tokenizer, MAX_LEN)
                        else:
                            train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
                            test_dataset  = TextClassificationDataset(test_texts, test_labels, tokenizer, MAX_LEN)

                        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                        total_steps = len(train_loader) * N_EPOCHS
                        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

                        early_stopper = EarlyStopping(patience=3)

                        # Training loop
                        for epoch in range(N_EPOCHS):
                            if VERBOSE >= 1:
                                print(f'Epoch {epoch+1}/{N_EPOCHS}')

                            if class_type == 'regress':
                                mse, rmse, mae, r2, t_loss_mean, t_loss_std, _ = train_regress(
                                    model, train_loader, optimizer, scheduler, device
                                )
                                if VERBOSE >= 2:
                                    print(f'Train: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, loss={t_loss_mean:.2f}±{t_loss_std:.2f}')
                                elif VERBOSE == 1:
                                    print(f'Mean train loss: {t_loss_mean:.2f} ± {t_loss_std:.2f}')
                                sys.stdout.flush()

                                val_mse, val_rmse, val_mae, val_r2, v_loss_mean, v_loss_std, _ = evaluate_regress(
                                    model, test_loader, device
                                )
                                if VERBOSE >= 2:
                                    print(f'Test: MSE={val_mse:.2f}, RMSE={val_rmse:.2f}, MAE={val_mae:.2f}, R2={val_r2:.2f}, loss={v_loss_mean:.2f}±{v_loss_std:.2f}\n')
                                elif VERBOSE == 1:
                                    print(f'Mean test loss: {v_loss_mean:.2f} ± {v_loss_std:.2f}')
                                sys.stdout.flush()

                                # Early stopping on negative MSE (lower MSE is better)
                                if early_stopper.step(-val_mse, model):
                                    print(f'Early stopping at epoch {epoch+1}')
                                    break
                            else:
                                # Classification
                                f1_train, prec_train, rec_train, conf_train, t_loss_mean, t_loss_std, _ = train(
                                    model, train_loader, optimizer, scheduler, device
                                )
                                if VERBOSE >= 2:
                                    print(f'Train: F1={f1_train:.2f}, Prec={prec_train:.2f}, Rec={rec_train:.2f}, loss={t_loss_mean:.2f}±{t_loss_std:.2f}')
                                elif VERBOSE == 1:
                                    print(f'Mean train loss: {t_loss_mean:.2f} ± {t_loss_std:.2f}')
                                sys.stdout.flush()

                                f1_val, prec_val, rec_val, conf_val, v_loss_mean, v_loss_std, preds, probs = evaluate(
                                    model, test_loader, device
                                )
                                if VERBOSE >= 2:
                                    print(f'Test: F1={f1_val:.2f}, Prec={prec_val:.2f}, Rec={rec_val:.2f}, loss={v_loss_mean:.2f}±{v_loss_std:.2f}\n')
                                elif VERBOSE == 1:
                                    print(f'Mean test loss: {v_loss_mean:.2f} ± {v_loss_std:.2f}')
                                sys.stdout.flush()

                                if early_stopper.step(f1_val, model):
                                    print(f'Early stopping at epoch {epoch+1}')
                                    break

                        # --- Save results after training ---
                        if class_type == 'regress':
                            print(f'Regression results: MSE={val_mse:.2f}, RMSE={val_rmse:.2f}, MAE={val_mae:.2f}, R2={val_r2:.2f}')
                            df_results.loc[len(df_results)] = [
                                asr, model_name, class_type, lr, q_consider,
                                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                round(val_mse, 2), round(val_rmse, 2), round(val_mae, 2), round(val_r2, 2)
                            ]
                        else:
                            # Build dataframe for majority voting
                            df_test_out = df_test[['dir_name', 'labels']].copy()
                            df_test_out['pred_label'] = preds
                            df_test_out['pred_proba'] = probs
                            df_test_out = df_test_out.rename(columns={'dir_name': 'r_IDs'})

                            f1, prec, rec, acc, conf, auc, spec, sens = majority_voting_pred_labels(
                                df_test_out, class_type, verbose=0
                            )
                            print(f'F1: {f1:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}')
                            print(f'Accuracy: {acc:.2f}, AUC: {auc:.2f}')
                            print('Confusion matrix:\n', conf)

                            df_results.loc[len(df_results)] = [
                                asr, model_name, class_type, lr, q_consider,
                                round(f1, 2), round(acc, 2), round(prec, 2), round(rec, 2),
                                conf, auc, round(spec, 2), round(sens, 2),
                                np.nan, np.nan, np.nan, np.nan
                            ]

                        df_results.to_csv(results_csv, index=False)

    print('\nALL EXPERIMENTS COMPLETED')
    elapsed = datetime.now() - start_time
    print(f'Finished at: {datetime.now().strftime("%Y/%m/%d at %H:%M:%S")}')
    print(f'Execution time: {elapsed}\n')