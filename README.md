# PROCESS-2: Reproducible Analysis Pipeline

This repository contains all scripts required to reproduce the preprocessing, statistical analyses, and baseline modelling experiments associated with the **PROCESS-2 speech dataset**.

The provided pipeline enables full reproducibility of the experiments reported in the accompanying publication.

---

## Repository Structure

```
PROCESS-2/
│
├── codes/                     # Main analysis and experiment scripts
│   ├── PROCESS2_gen_audio_info.py
│   ├── PROCESS2_data_analysis.py
│   ├── PROCESS2_rename_cogno.py
│   ├── PROCESS2_embed.py
│   ├── PROCESS2_gen_ASR.py
│   ├── PROCESS2_BASELINE_class.py
│   └── PROCESS2_BASELINE_LLM.py
│
├── envs/                      # Conda environment configuration files
│   ├── aconda.yml
│   ├── embed.yml
│   └── torch-gpu.yml
│
├── results/
│   └── logs/                  # Automatically generated experiment logs
│
└── README.md
```

---

## Installation

We recommend using **Conda** to ensure environment reproducibility.

Create environments using:

```bash
conda env create -f envs/<environment>.yml
```

### Available Environments

| Environment   | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| **ACONDA**    | Data preprocessing, statistics, classical machine learning |
| **embed**     | Speech embedding extraction                                |
| **torch-gpu** | ASR generation and LLM experiments (GPU required)          |

Activate an environment before running each step:

```bash
conda activate <environment_name>
```

---

## Reproducible Workflow

Run the following scripts **in order** to reproduce all analyses and baseline experiments.

All commands assume execution from the `codes/` directory unless stated otherwise.

---

### Step 1 — Generate Audio Metadata

Computes recording statistics including:

* audio duration
* signal-to-noise ratio (SNR)
* file size statistics

```bash
python PROCESS2_gen_audio_info.py
```

Environment: **ACONDA**

---

### Step 2 — Statistical Data Analysis

Reproduces descriptive dataset statistics reported in the paper.

```bash
python PROCESS2_data_analysis.py
```

Environment: **ACONDA**

---

### Step 2A — Dataset Utilities

Utility script used for dataset organisation and filename harmonisation.

```bash
python PROCESS2_rename_cogno.py
```

Environment: **ACONDA**

---

### Step 3 — Speech Embedding Extraction

Generates acoustic and linguistic embeddings used for downstream modelling.

```bash
python PROCESS2_embed.py
```

Environment: **embed**

---

### Step 4 — Automatic Speech Recognition (ASR)

Generates automatic transcripts from audio recordings.

```bash
python PROCESS2_gen_ASR.py
```

Environment: **torch-gpu**

---

### Step 5 — Classical Baseline Models

Run from the `codes/` directory:

```bash
python PROCESS2_BASELINE_class.py 12 |& tee -a ../results/logs/PROCESS2_BASELINE_class.txt
```

`12` specifies the number of CPU cores used.

Environment: **ACONDA**

---

### Step 6 — LLM Baseline Models

Run from the `codes/` directory:

```bash
python PROCESS2_BASELINE_LLM.py 0,1,2,3 |& tee -a ../results/logs/PROCESS2_BASELINE_LLM.txt
```

`0,1,2,3` correspond to GPU device IDs.

Environment: **torch-gpu**

---

## Outputs

Running the full pipeline produces:

* audio metadata summaries
* statistical dataset analyses
* speech embeddings
* ASR transcripts
* baseline classifier and regressor outputs
* experiment logs

Logs are automatically written to:

```
results/logs/
```

---

## Reproducibility

All preprocessing, analysis, and modelling steps reported in the PROCESS-2 dataset publication can be reproduced using this repository.

Researchers may adapt or extend the pipeline for benchmarking, comparative modelling, or future speech-based cognitive assessment studies.

---

## Citation

If you use the PROCESS-2 dataset or associated code, please cite the corresponding dataset publication.

[Insert Citation / BibTeX here]
