# 🚢 HullInspect — Maritime Hull Defect Detection & Reporting

A web-based system for detecting and reporting hull defects from inspection images.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Clone the Repository](#clone-the-repository)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Run the Server](#run-the-server)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Make sure the following are installed before you start:

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Runtime |
| **Git LFS** | 3.0+ | Large files (model weights, images) |
| **Ollama** | 0.1.0+ | Local LLM for chatbot |

### Install Git LFS

Git LFS is **required** — the repo stores model weights (`.pt`) and images via LFS. Without it you'll get pointer files instead of actual data.

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Windows — download from https://git-lfs.com
```

After installing, run once:

```bash
git lfs install
```

### Install Ollama

Download from [ollama.com](https://ollama.com/) or:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

Then pull the Phi-3 Mini model:

```bash
ollama pull phi3:mini
```

---

## Clone the Repository

```bash
git clone git@collaborating.tuhh.de:cvv4692/research-project.git
cd research-project
```

> **Note:** If you cloned before installing Git LFS, run `git lfs pull` to download the actual files.

---

## Setup

### 1. Create a virtual environment

```bash
cd hull_inspection
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run database migrations

```bash
python manage.py migrate
```

---

## Environment Variables

Create a `.env` file in the `hull_inspection/` directory (or copy the example):

```bash
cp .env.example .env        # macOS / Linux
# copy .env.example .env    # Windows
```

Then edit `.env` with your API keys:

```env
# GPT-4o-mini via GitHub Models (primary VLM)
GITHUB_TOKEN=your_github_token
GITHUB_MODELS_ENDPOINT=https://models.inference.ai.azure.com
GITHUB_MODELS_MODEL=gpt-4o-mini

# Gemini 3.1 Flash Lite via OpenRouter (fallback VLM)
OPENROUTER_API_KEY=your_openrouter_key
```

| Variable | Where to get it |
|----------|----------------|
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) — enable GitHub Models access |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) — free tier available |

> **Without API keys the app still works** — it falls back to template-based reports. The keys enable AI-powered VLM analysis.

---

## Run the Server

### 1. Start Ollama (for the chatbot)

```bash
ollama serve
```

Leave this running in a separate terminal.

### 2. Start Django

```bash
cd hull_inspection
python manage.py runserver 8090
```

Open **http://localhost:8090** in your browser.

### What you can do

- 📤 **Upload** hull images for inspection
- 🔍 **Choose** detection method — YOLO11-Seg or Pruned Eigen-Tree
- 🖼️ **View** annotated images with detected defect regions
- 📋 **Read** structured survey reports (cross-verified by VLM)
- 📄 **Download** PDF reports
- 💬 **Chat** with the inspection assistant about past results
- 📂 **Browse** inspection history

---

## Project Structure

```
hull_inspection/
├── manage.py                   # Django entry point
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── inspection/
│   ├── models.py               # Database models
│   ├── views.py                # Request handlers
│   └── services/
│       ├── yolo_seg_inference.py    # YOLO11-Seg detection
│       ├── eigen_tree_service.py    # Eigen-Tree detection
│       ├── vlm_service_v2.py        # VLM report generation
│       ├── chatbot_service.py       # Ollama chatbot
│       └── report_generator.py      # PDF generation
├── ml_models/yolo/
│   ├── weights/best_seg.pt     # Trained model weights (Git LFS)
│   ├── train_seg.py            # Segmentation training script
│   └── config/                 # Training configs
├── datasets/                   # MaVeCoDD-derived datasets
├── templates/                  # HTML templates (Tailwind CSS)
└── media/                      # Uploaded images & generated reports
```

---

## Troubleshooting

**Model weights are tiny text files (pointer files):**
```bash
git lfs pull
```

**Ollama chatbot not responding:**
```bash
# Make sure Ollama is running
ollama serve

# Verify the model is pulled
curl http://localhost:11434/api/tags
```

**VLM report says "Template fallback":**
Check that `.env` has a valid `GITHUB_TOKEN`. Without it, the system still works but uses simpler report generation.

**Port already in use:**
```bash
lsof -ti:8090 | xargs kill -9
python manage.py runserver 8090
```

**`ModuleNotFoundError`:**
```bash
pip install -r requirements.txt
```

---

