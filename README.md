<div align="center">

# 🧪 PROCESS-2  
### Reproducible Speech Analysis & Baseline Modelling Pipeline

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Conda](https://img.shields.io/badge/Environment-Conda-green.svg)]()
[![GPU](https://img.shields.io/badge/GPU-Optional-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/CognoSpeak/PROCESS-2)
[![Reproducibility](https://img.shields.io/badge/Reproducible-Research-brightgreen.svg)]()
[![Open Science](https://img.shields.io/badge/Open%20Science-Compatible-blueviolet.svg)]()
[![DOI](https://zenodo.org/badge/1197188333.svg)](https://doi.org/10.5281/zenodo.19900224)

**Official analysis and experiment pipeline for the PROCESS-2 speech dataset**

</div>

---

## 🔬 Overview

This repository contains the complete **reproducible analysis pipeline** for the **PROCESS-2 speech dataset**, publicly available on Hugging Face:

👉 https://huggingface.co/datasets/CognoSpeak/PROCESS-2

The pipeline reproduces:

- dataset preprocessing
- statistical analyses
- speech feature extraction
- automatic speech recognition (ASR)
- baseline machine learning models
- large language model (LLM) experiments

All experiments reported in the accompanying publication can be reproduced directly from this repository.

---

## 🧬 Research Context

PROCESS-2 supports research in:

- speech-based cognitive assessment
- dementia and MCI biomarkers
- clinical speech analytics
- multimodal AI evaluation
- reproducible machine learning pipelines

The repository follows **open-science and reproducibility best practices**, enabling transparent benchmarking and future comparative studies.

---

## 📂 Repository Structure

```
PROCESS-2/
│
├── codes/                     # Analysis & modelling scripts
│   ├── PROCESS2_gen_audio_info.py
│   ├── PROCESS2_data_analysis.py
│   ├── PROCESS2_embed.py
│   ├── PROCESS2_gen_ASR.py
│   ├── PROCESS2_BASELINE_class.py
│   └── PROCESS2_BASELINE_LLM.py
│
├── envs/                      # Conda environments
│   ├── aconda.yml
│   ├── embed.yml
│   └── torch-gpu.yml
│
├── results/
│   └── logs/                  # Automatically generated logs
│
└── README.md
```

---

## ⚙️ Installation

We strongly recommend **Conda** for full reproducibility.

### Create environments

```bash
conda env create -f envs/<environment>.yml
```

### Available Environments

| Environment | Purpose |
|-------------|---------|
| **ACONDA** | Preprocessing, statistics, classical ML |
| **embed** | Speech embedding extraction |
| **torch-gpu** | ASR generation & LLM experiments |

Activate before running each stage:

```bash
conda activate <environment_name>
```

---

## 🚀 Reproducible Workflow

Run scripts **in order** to reproduce all results.

All commands assume execution from:

```
codes/
```

---

### Step 1 — Generate Audio Metadata

Computes recording statistics:

- duration
- signal-to-noise ratio (SNR)
- file size summaries

```bash
python PROCESS2_gen_audio_info.py
```

Environment: **ACONDA**

---

### Step 2 — Statistical Dataset Analysis

Reproduces descriptive statistics reported in the paper.

```bash
python PROCESS2_data_analysis.py
```

Environment: **ACONDA**

---

### Step 3 — Speech Embedding Extraction

Generates acoustic and linguistic representations for modelling.

```bash
python PROCESS2_embed.py
```

Environment: **embed**

---

### Step 4 — Automatic Speech Recognition (ASR)

Produces automatic transcripts.

```bash
python PROCESS2_gen_ASR.py
```

Environment: **torch-gpu**

GPU recommended but optional depending on configuration.

---

### Step 5 — Classical Baseline Models

```bash
python PROCESS2_BASELINE_class.py 12 |& tee -a ../results/logs/PROCESS2_BASELINE_class.txt
```

`12` = number of CPU cores.

Environment: **ACONDA**

---

### Step 6 — LLM Baseline Experiments

```bash
python PROCESS2_BASELINE_LLM.py 0,1,2,3 |& tee -a ../results/logs/PROCESS2_BASELINE_LLM.txt
```

`0,1,2,3` correspond to GPU device IDs.

Environment: **torch-gpu**

---

## 📊 Outputs

Running the full pipeline produces:

- audio metadata summaries
- dataset statistical analyses
- speech embeddings
- ASR transcripts
- baseline model predictions
- experiment evaluation outputs
- execution logs

Logs are automatically stored in:

```
results/logs/
```

---

## 🔁 Reproducibility Statement

This repository provides **full computational reproducibility** for the PROCESS-2 dataset publication.

The workflow ensures:

- deterministic preprocessing
- version-controlled environments
- explicit experiment ordering
- logged execution traces

Researchers may reuse or extend the pipeline for:

- benchmarking studies
- model comparison
- replication experiments
- clinical speech AI research

---

## 🤗 Dataset Access

The PROCESS-2 dataset is hosted on Hugging Face:

👉 https://huggingface.co/datasets/CognoSpeak/PROCESS-2

Please follow dataset licensing and ethical usage guidelines.

---

## 📚 Citation

If you use **PROCESS-2**, please cite **the dataset paper, and code repository**.

---

### 📄 Dataset Paper

Pahar, M., Illingworth, C. H., Mirheidari, B., Elghazaly, H., Peters, F., Young, S., Leung, W. Z., Kaur, L., Blackburn, D., & Christensen, H. (2026).  
**PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection.**  
*arXiv preprint arXiv:2605.14888.*

#### BibTeX
```bibtex
@article{pahar2026PROCESS2,
  title={{PROCESS}-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection},
  author={Pahar, Madhurananda and Illingworth, Caitlin H. and Mirheidari, Bahman and Elghazaly, Hend and Peters, Fritz and Young, Sophie and Leung, Wing-Zin and Kaur, Labhpreet and Blackburn, Daniel and Christensen, Heidi},
  journal={arXiv preprint arXiv:2605.14888},
  year={2026}
}
```

### 📄 Code repository

Pahar, M. (2026) *Madhurananda/PROCESS-2: PROCESS-2*. Zenodo. doi:10.5281/zenodo.19900225.

#### BibTeX

```bibtex
@software{pahar_PROCESS2_codes,
  author       = {Pahar, Madhurananda},
  title        = {Madhurananda/{PROCESS}-2: {PROCESS}-2},
  month        = Apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.19900225},
  url          = {https://doi.org/10.5281/zenodo.19900225}
}
```

---

## 🔓 Open Science & Ethics

This repository supports transparent and ethical AI research.

Included:
- ✅ code
- ✅ experiment pipeline
- ✅ reproducibility infrastructure

Not included:
- ❌ private participant data

All dataset usage must comply with ethical approval and data governance policies.

---

<div align="center">

**Reproducible Speech AI Research • Open Science • Clinical Machine Learning**

</div>
