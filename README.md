# DSCD: Dynamic Safety Contrastive Decoding for LLM Detoxification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## Overview

**DSCD (Dynamic Safety Contrastive Decoding)** is a novel decoding strategy for detoxifying Large Language Models (LLMs). It combines early-exit layer contrastive decoding with the DINM (Detoxifying In-Network Modulation) editing method to enhance LLM safety while preserving generation quality.

### Key Features

- **Early Exit Layer Contrastive Decoding**: Leverages outputs from multiple transformer layers to contrast safe vs. toxic generation patterns
- **DINM Integration**: Combines with DINM model editing for enhanced detoxification
- **Flexible Layer Selection**: Supports configurable premature and mature layer combinations
- **Comprehensive Evaluation**: Includes safety classification and fluency metrics

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DSCD Framework                        │
├─────────────────────────────────────────────────────────┤
│  Input Prompt                                            │
│       ↓                                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Modified LLaMA Model                   │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  Early Exit Layers (Premature Layers)   │    │    │
│  │  │  → Layer 0, 2, 4, ... (configurable)    │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  Mature Layer (Final Layer)             │    │    │
│  │  │  → Layer 32 (default for LLaMA-7B)      │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│       ↓                                                  │
│  Contrastive Decoding: P(safe) - P(toxic)               │
│       ↓                                                  │
│  Safe Output Generation                                  │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.1
- CUDA >= 11.7 (for GPU support)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/DSCD.git
cd DSCD
```

2. **Create conda environment**
```bash
conda create -n dscd python=3.8
conda activate dscd
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

> ⚠️ **Important**: This project uses a **modified local version of transformers**. Do NOT install transformers from pip. The local `transformers/` directory contains necessary modifications for early-exit layer support.

4. **Download required models**

- **LLaMA-2-7B-Chat**: Download from [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- **Safety Classifier**: `zjunlp/SafeEdit-Safety-Classifier`

## Project Structure

```
DSCD/
├── run_dscd.py                 # Main entry point
├── easyeditor/                 # EasyEdit framework
├── transformers/               # Modified transformers (local)
│   ├── generation/
│   │   └── utils.py           # Modified greedy_search with DSCD
│   └── models/llama/
│       └── modeling_llama.py  # Added early_exit_layers support
├── hparams/                    # Hyperparameter configs
│   └── llama-7b.yaml          # LLaMA-7B configuration
├── data/                       # Dataset directory
│   └── SafeEdit/              # SafeEdit dataset
├── requirements.txt            # Python dependencies
└── README_DSCD.md             # This file
```

## Usage

### Basic Usage

```bash
python run_dscd.py \
    --editing_method=DINM \
    --edited_model=llama-2-7b-chat \
    --hparams_dir=./hparams/llama-7b.yaml \
    --safety_classifier_dir=zjunlp/SafeEdit-Safety-Classifier \
    --metrics_save_dir=./results \
    --early-exit-layers="0,2,4,6,8,10,12,14,32"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--editing_method` | Editing method (DINM/MEND) | Required |
| `--edited_model` | Model name for logging | Required |
| `--hparams_dir` | Path to hyperparameter config | Required |
| `--safety_classifier_dir` | Safety classifier model path | Required |
| `--data_dir` | Dataset directory | `./data` |
| `--metrics_save_dir` | Output directory | `./safety_results` |
| `--early-exit-layers` | Comma-separated layer indices | `-1` |
| `--repetition_penalty` | Repetition penalty | `1.7` |
| `--relative_top` | Relative top filtering | `0.1` |

### Early Exit Layers Configuration

The `--early-exit-layers` parameter specifies which layers to use for contrastive decoding:
- **Premature layers**: All layers except the last one in the list
- **Mature layer**: The last layer in the list

Example: `"0,2,4,6,8,10,12,14,32"` means:
- Premature layers: 0, 2, 4, 6, 8, 10, 12, 14
- Mature layer: 32

## Core Modifications

### 1. LLaMA Model (`transformers/models/llama/modeling_llama.py`)

Added `early_exit_layers` parameter to the forward function:

```python
def forward(
    self,
    ...
    early_exit_layers: Optional[List[int]] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Returns logits from specified layers
    if early_exit_layers is not None:
        logits_dict = {}
        for early_exit_layer in early_exit_layers:
            logits = self.lm_head(outputs.hidden_states[early_exit_layer])
            logits_dict[early_exit_layer] = logits
        return logits_dict, final_outputs
```

### 2. Greedy Search (`transformers/generation/utils.py`)

Modified `greedy_search` to implement DSCD:

```python
def greedy_search(
    self,
    dscd,                              # Enable DSCD mode
    input_ids,
    mature_layer,                      # Final layer index
    candidate_premature_layers,        # Early exit layers
    relative_top,                      # Filtering threshold
    ...
):
    # Contrastive decoding logic
    # P_final = P_mature - α * P_premature
```

## Evaluation Metrics

The framework evaluates:

| Metric | Description |
|--------|-------------|
| **DS** | Defense Success rate |
| **DG_onlyQ** | Defense Generalization (question only) |
| **DG_otherA** | Defense Generalization (other answers) |
| **DG_otherQ** | Defense Generalization (other questions) |
| **DG_otherAQ** | Defense Generalization (other Q&A) |
| **Fluency** | N-gram entropy based fluency |

## Results

Results are saved to the specified `--metrics_save_dir`:

- `DINM_ORI vs DINM_NEW.json`: Detailed per-sample results
- `*_performance_avg.json`: Aggregated performance metrics
- `all_run_times_*.json`: Runtime statistics for each method

## Citation

If you use this code, please cite:

```bibtex
@article{dscd2024,
  title={DSCD: Dynamic Safety Contrastive Decoding for LLM Detoxification},
  author={Your Name},
  year={2024}
}
```

## Acknowledgments

This project builds upon:
- [EasyEdit](https://github.com/zjunlp/EasyEdit) - Knowledge editing framework
- [SafeEdit](https://huggingface.co/datasets/zjunlp/SafeEdit) - Safety editing dataset
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
