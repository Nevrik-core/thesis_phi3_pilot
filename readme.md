# Phi-4-mini Quantization Study

This repository contains the code and final experimental artifacts for the thesis:

**"Impact of Quantization on Quality and Resource Efficiency of Local Phi-4-mini Inference"**

## Scope

The study evaluates **Phi-4-mini-instruct** in local CPU-only inference using **Ollama** and **GGUF** model artifacts.

The main goal is to measure how quantization affects:

- answer quality;
- memory usage;
- inference speed;
- token throughput;
- behavioral stability on alignment-style tasks;
- differences between Ukrainian and English benchmark performance.

## Quantization modes

The final experiment compares the following modes:

- BF16
- Q8_0
- Q6_K
- Q5_K_M
- Q4_K_M

BF16 is used as a high-precision GGUF reference. Q8_0, Q6_K, Q5_K_M and Q4_K_M form the main quantization curve.

## Benchmarks

The final evaluation uses three benchmark groups:

- **QA**
  - UA-SQuAD for Ukrainian extractive question answering
  - SQuAD for English extractive question answering

- **BELEBELE**
  - `ukr_Cyrl`
  - `eng_Latn`

- **UAlign**
  - ETHICS
  - Social Chemistry 101
  - Ukrainian and English parallel examples

## Final results

Final results are stored in:

```text
results/v4/
```

This directory contains the final CSV summaries, detailed outputs, analysis tables and generated figures used in the thesis.

The most important result categories are:

- benchmark summaries;
- QA strict and content-adjusted analysis;
- BELEBELE accuracy results;
- UAlign accuracy and behavior analysis;
- speed and token-efficiency analysis;
- memory and quality/resource trade-off figures.

## Repository structure

Recommended structure:

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── dataset_loaders.py
│   ├── belebele_loader.py
│   ├── ualign_loader.py
│   ├── ollama_runner.py
│   ├── resource_monitor.py
│   ├── eval_utils.py
│   ├── pilot_qa_eval.py
│   ├── pilot_belebele_eval.py
│   ├── pilot_ualign_eval.py
│   ├── run_final_matrix.py
│   ├── analyze_qa_content_accuracy.py
│   ├── analyze_speed_tokens.py
│   ├── analyze_ualign_behavior.py
│   ├── plot_ualign_behavior.py
│   └── plot_ualign_orientation_curves.py
├── results/
│   └── v4/
└── legacy/
```

The `legacy/` directory may contain older pilot scripts, smoke tests or intermediate experiments that were useful during development but are not part of the final clean pipeline.

## Requirements

Python dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

The project also requires **Ollama** to be installed separately.

## Ollama setup

The experiments use local Ollama inference through the HTTP API:

```text
http://localhost:11434/api/chat
```

For CPU-only evaluation, start Ollama with GPU disabled.

PowerShell example:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
ollama serve
```

The benchmark runner pulls the required Hugging Face GGUF models through Ollama when needed.

## Model sources

The experiment uses GGUF models from Hugging Face through Ollama model identifiers, including:

```text
hf.co/professorf/Phi-4-mini-instruct-gguf:BF16
hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0
hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q6_K
hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q5_K_M
hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M
```

The exact active model registry is defined in:

```text
src/config.py
```

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama in CPU-only mode:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
ollama serve
```

Run the final benchmark matrix:

```bash
python src/run_final_matrix.py
```

The final matrix runs:

- QA evaluation;
- BELEBELE evaluation;
- UAlign evaluation;
- model switching through `ACTIVE_OLLAMA_MODEL_KEY`;
- isolated CPU-only Ollama execution;
- resource monitoring for client process, model process and system memory.

## Running individual evaluations

QA:

```bash
python src/pilot_qa_eval.py
```

BELEBELE:

```bash
python src/pilot_belebele_eval.py
```

UAlign:

```bash
python src/pilot_ualign_eval.py
```

The active model can be selected with:

```powershell
$env:ACTIVE_OLLAMA_MODEL_KEY="bartowski_q4_k_m"
python src/pilot_qa_eval.py
```

Available model keys are defined in `src/config.py`.

## Running additional analysis

Speed and token-efficiency analysis:

```bash
python src/analyze_speed_tokens.py --input-dir results/v4 --version v4
```

UAlign behavior analysis:

```bash
python src/analyze_ualign_behavior.py --input-dir results/v4 --version v4
```

QA content-adjusted analysis:

```bash
python src/analyze_qa_content_accuracy.py --input-dir results/v4 --version v4
```

UAlign behavior plots:

```bash
python src/plot_ualign_behavior.py --input-dir results/v4/ualign_behavior_analysis --version v4
```

UAlign orientation curves:

```bash
python src/plot_ualign_orientation_curves.py --input results/v4/ualign_behavior_analysis/ualign_orientation_summary_v4.csv --output-dir results/v4/ualign_orientation_figures --version v4
```

## Notes on result interpretation

QA results are reported with strict EM/F1 metrics. For Ukrainian QA, an additional content-adjusted analysis is included because strict string matching can underestimate semantically correct Ukrainian answers due to inflection, formatting differences or extra text.

BELEBELE is treated as a multiple-choice reading comprehension benchmark. The script normalizes Latin and visually similar Cyrillic answer letters in Ukrainian prompts to avoid counting clear formatting variants as invalid answers.

UAlign is analyzed not only by aggregate accuracy, but also by predicted label distributions, label flips between quantization modes and EN-UA consistency on parallel examples. This is important because accuracy alone may hide behavioral shifts.

## Important limitations

The results describe a specific local inference setup:

- Phi-4-mini-instruct;
- GGUF model artifacts;
- Ollama runtime;
- CPU-only execution;
- fixed prompts;
- deterministic generation settings;
- selected Ukrainian and English benchmark subsets.

The results should not be interpreted as universal conclusions for all LLMs, all hardware setups or all quantization methods.

## Deterministic generation settings

The final experiments use deterministic generation:

```text
temperature = 0.0
do_sample = False
num_ctx = 2048
```

For QA:

```text
max_new_tokens = 32
```

For multiple-choice tasks:

```text
max_new_tokens = 4
```

## Author

Andrii Ilin

Master's thesis project, GoIT Neoversity.
