#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESS2_gen_ASR.py

Generate Automatic Speech Recognition (ASR) transcripts for all audio files in the
PROCESS‑2 dataset using two models:
- Whisper (medium) – works with original sample rate (44.1 kHz)
- Wav2Vec2 (facebook/wav2vec2-base-960h) – requires 16 kHz resampling

For each audio file, the script produces:
- Plain text transcript (.txt)
- Pickled dictionary with full model output (.dict)
- Word‑level timestamps (CSV)
- (Wav2Vec2 only) Character‑level timestamps (CSV)

Author: Madhurananda Pahar
Date: 8th APR, 2026
"""

import os
import sys
import time
import pickle
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import librosa
from tqdm import tqdm

import whisper
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC


# -----------------------------------------------------------------------------
# 1. Utility Functions 
# -----------------------------------------------------------------------------

def write_file(file_name, to_write):
    """Write a string to a text file (overwrites if exists)."""
    with open(file_name, "w") as f:
        f.write(to_write)


def save_file(file_save_name, file_data):
    """Pickle and save any Python object to a file."""
    with open(file_save_name, "wb") as handle:
        pickle.dump(file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_load_name):
    """Load a pickled Python object from a file."""
    with open(file_load_name, "rb") as f:
        return pickle.load(f)


def gen_whisper_word_timestamps(trans_dict, time_word_out):
    """
    Extract word‑level timestamps from Whisper's output dictionary
    and save as a CSV file with columns: word, start, end, probs.
    """
    df = pd.DataFrame(columns=['word', 'start', 'end', 'probs'])
    for segment in trans_dict['segments']:
        for word_info in segment['words']:
            df.loc[len(df)] = [
                word_info['word'],
                word_info['start'],
                word_info['end'],
                word_info['probability']
            ]
    df.to_csv(time_word_out, index=False)


def gen_w2v2_word_timestamps(outputs, time_word_out, time_offset):
    """
    Extract word‑level timestamps from Wav2Vec2's output object
    and save as CSV. `time_offset` is computed from model config.
    """
    df = pd.DataFrame(columns=['word', 'start', 'end'])
    for w in outputs.word_offsets:
        df.loc[len(df)] = [
            w["word"],
            w["start_offset"] * time_offset,
            w["end_offset"] * time_offset
        ]
    df.to_csv(time_word_out, index=False)


def gen_w2v2_char_timestamps(outputs, time_char_out, time_offset):
    """
    Extract character‑level timestamps from Wav2Vec2's output object
    and save as CSV.
    """
    df = pd.DataFrame(columns=['char', 'start', 'end'])
    for c in outputs.char_offsets:
        df.loc[len(df)] = [
            c["char"],
            c["start_offset"] * time_offset,
            c["end_offset"] * time_offset
        ]
    df.to_csv(time_char_out, index=False)


# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__' and '__file__' in globals():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    data_dir = '../data/'
    asr_log_dir = '../data/ASR_logs/'
    os.makedirs(asr_log_dir, exist_ok=True)

    # Metadata file
    df_meta = pd.read_csv(data_dir + 'meta-info.csv').rename(columns={'IDs': 'dir_name'})

    # Device for Whisper (Wav2Vec2 will also use GPU if available, but we keep separate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2.1 Whisper (Medium) Transcription
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Running Whisper Medium ASR ...")
    print("="*60)
    sys.stdout.flush()

    # Load Whisper model
    model_whisper = whisper.load_model("medium").to(device)

    # Error log
    error_csv = asr_log_dir + f'ASR_errors__{datetime.today().strftime("%Y_%m_%d__%H_%M_%S")}.csv'
    df_errors = pd.DataFrame(columns=['dir_name', 'audio', 'ASR', 'comment'])

    for dir_name in tqdm(df_meta.dir_name, desc="Whisper"):
        audio_dir = os.path.join(data_dir, dir_name)
        wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))

        for wav_path in wav_files:
            base = wav_path.replace('.wav', '')
            txt_out = base + '__ASR_Whisper.txt'
            dict_out = base + '__ASR_Whisper.dict'
            word_csv = base + '__ASR_Whisper__WORD.csv'

            # Skip if all outputs already exist
            if os.path.exists(txt_out) and os.path.exists(dict_out) and os.path.exists(word_csv):
                continue

            # If dictionary exists but some files missing, regenerate from dict
            if os.path.exists(dict_out):
                whisper_dict = load_file(dict_out)
                if not os.path.exists(word_csv):
                    try:
                        gen_whisper_word_timestamps(whisper_dict, word_csv)
                    except Exception as e:
                        print(f"Error generating word timestamps from dict: {e}")
                        # Continue to full re‑run? We'll just log and skip.
                if not os.path.exists(txt_out):
                    write_file(txt_out, whisper_dict['text'])
                continue

            # Otherwise run Whisper from scratch
            try:
                transcript = model_whisper.transcribe(
                    audio=wav_path,
                    language="en",
                    word_timestamps=True
                )
                save_file(dict_out, transcript)
                gen_whisper_word_timestamps(transcript, word_csv)
                write_file(txt_out, str(transcript['text']))
            except Exception as e:
                print(f"Whisper failed for {wav_path}: {e}")
                # Log error
                if os.path.exists(error_csv):
                    df_errors = pd.read_csv(error_csv)
                df_errors.loc[len(df_errors)] = [dir_name, wav_path, 'Whisper_medium', str(e)]
                df_errors.to_csv(error_csv, index=False)

    print("Whisper transcription completed successfully.")
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # 2.2 Wav2Vec2 Transcription (requires 16 kHz audio)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Running Wav2Vec2 ASR ...")
    print("="*60)
    sys.stdout.flush()

    # Load Wav2Vec2 model and tokenizer
    model_w2v2 = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer_w2v2 = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    SR_W2V2 = 16000  # required sample rate

    for dir_name in tqdm(df_meta.dir_name, desc="Wav2Vec2"):
        audio_dir = os.path.join(data_dir, dir_name)
        wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))

        for wav_path in wav_files:
            base = wav_path.replace('.wav', '')
            txt_out = base + '__ASR_W2V2.txt'
            dict_out = base + '__ASR_W2V2.dict'
            word_csv = base + '__ASR_W2V2__WORD.csv'
            char_csv = base + '__ASR_W2V2__CHAR.csv'

            # Skip if all outputs already exist
            if (os.path.exists(txt_out) and os.path.exists(dict_out) and
                os.path.exists(word_csv) and os.path.exists(char_csv)):
                continue

            # If dictionary exists, regenerate missing files
            if os.path.exists(dict_out):
                w2v2_dict = load_file(dict_out)
                # Compute time_offset (needed for timestamp functions)
                time_offset = (model_w2v2.config.inputs_to_logits_ratio /
                               feature_extractor.sampling_rate)
                if not os.path.exists(word_csv):
                    gen_w2v2_word_timestamps(w2v2_dict, word_csv, time_offset)
                if not os.path.exists(char_csv):
                    gen_w2v2_char_timestamps(w2v2_dict, char_csv, time_offset)
                if not os.path.exists(txt_out):
                    write_file(txt_out, w2v2_dict.text)
                continue

            # Otherwise run Wav2Vec2 from scratch
            try:
                # Load and resample audio to 16 kHz
                speech, _ = librosa.load(wav_path, sr=SR_W2V2)

                # Prepare input
                input_values = feature_extractor(
                    speech, return_tensors="pt", sampling_rate=SR_W2V2
                ).input_values

                # Forward pass
                with torch.no_grad():
                    logits = model_w2v2(input_values).logits[0]
                pred_ids = torch.argmax(logits, axis=-1)

                # Decode with word/char offsets
                outputs = tokenizer_w2v2.decode(
                    pred_ids,
                    output_word_offsets=True,
                    output_char_offsets=True
                )

                # Compute time offset
                time_offset = (model_w2v2.config.inputs_to_logits_ratio /
                               feature_extractor.sampling_rate)

                # Save outputs
                write_file(txt_out, outputs.text)
                save_file(dict_out, outputs)
                gen_w2v2_char_timestamps(outputs, char_csv, time_offset)
                gen_w2v2_word_timestamps(outputs, word_csv, time_offset)

            except Exception as e:
                print(f"Wav2Vec2 failed for {wav_path}: {e}")
                if os.path.exists(error_csv):
                    df_errors = pd.read_csv(error_csv)
                df_errors.loc[len(df_errors)] = [dir_name, wav_path, 'Wav2Vec2', str(e)]
                df_errors.to_csv(error_csv, index=False)

    print("Wav2Vec2 transcription completed successfully.")
    sys.stdout.flush()
    
    
    
    # -------------------------------------------------------------------------
    # 2.3 WER Calculation using jiwer (compatible with older jiwer)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing Word Error Rate (WER) ...")
    print("=" * 60)
    sys.stdout.flush()
    
    import jiwer
    from jiwer import wer, process_words
    import string
    
    def normalize(text: str) -> str:
        """
        Apply same transformations: lowercase, remove punctuation,
        collapse multiple spaces, strip.
        Returns the cleaned text string.
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation (including apostrophes, hyphens, etc.)
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Collapse multiple spaces
        text = ' '.join(text.split())
        return text
    
    results = []
    for dir_name in tqdm(df_meta.dir_name, desc="WER"):
        audio_dir = os.path.join(data_dir, dir_name)
        wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
        for wav_path in wav_files:
            base = wav_path.replace('.wav', '')
            ref_path = base + '.txt'
            hyp_whisper = base + '__ASR_Whisper.txt'
            hyp_w2v2 = base + '__ASR_W2V2.txt'
    
            if not os.path.exists(ref_path):
                continue
    
            try:
                ref_raw = open(ref_path, 'r', encoding='utf-8').read().strip()
                reference = normalize(ref_raw)
            except Exception as e:
                print(f"Error reading {ref_path}: {e}")
                continue
    
            for model_name, hyp_path in [('Whisper', hyp_whisper), ('Wav2Vec2', hyp_w2v2)]:
                if not os.path.exists(hyp_path):
                    continue
                try:
                    hyp_raw = open(hyp_path, 'r', encoding='utf-8').read().strip()
                    hypothesis = normalize(hyp_raw)
                except Exception as e:
                    print(f"Error reading {hyp_path}: {e}")
                    continue
    
                # Now call wer and process_words with the normalized strings
                error_rate = wer(reference, hypothesis)
                measures = process_words(reference, hypothesis)
    
                results.append({
                    'dir_name': dir_name,
                    'audio': os.path.basename(wav_path),
                    'model': model_name,
                    'WER': round(error_rate, 4),
                    'substitutions': measures.substitutions,
                    'deletions': measures.deletions,
                    'insertions': measures.insertions,
                    'hits': measures.hits,
                    'reference': ref_raw,   # keep original for inspection
                    'hypothesis': hyp_raw
                })
    
    # Save results
    wer_csv = os.path.join(asr_log_dir, 'WER_results.csv')
    df_wer = pd.DataFrame(results)
    df_wer.to_csv(wer_csv, index=False)
    
    summary = df_wer.groupby('model')['WER'].mean()
    print("\nAverage WER per model:")
    print(summary.to_string())
    print(f"\nDetailed results saved to {wer_csv}")
    sys.stdout.flush()
    
    
    
    