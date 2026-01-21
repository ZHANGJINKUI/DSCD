# DSCD: Dynamic Safety Contrastive Decoding for LLM Detoxification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## Overview

**DSCD (Dynamic Safety Contrastive Decoding)** 
Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work innovatively proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLMs detoxification without parameter fine-tuning. DSCD strengthens the inner token distribution of the safety layer while weakening that of hallucination and toxic layer during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD’s effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD’s potential as a practical and scalable solution for safer LLM deployments.

### Key Features

- **Early Exit Layer Contrastive Decoding**: Leverages outputs from multiple transformer layers—including safety, toxicity, and hallucination layers—to contrast safe vs. toxic generation patterns and mitigate nonsensical outputs.
- **DINM Integration**: Combines with DINM model editing for enhanced detoxification
- **Flexible Layer Selection**: Supports configurable premature and mature layer combinations
- **Comprehensive Evaluation**: Includes safety classification and fluency metrics

## Architecture

```
<div align="center">
  <img src="https://raw.githubusercontent.com/ZHANGJINKUI/DSCD/main/figures/DSCD_architecture.png" alt="DSCD Framework" style="width: 80%; max-width: 800px;">
</div>
```

## Installation

### Requirements

- Python >= 3.10
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
conda create -n dscd python=3.10
conda activate dscd
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

> ⚠️ **Important**: This project uses a **modified local version of transformers**. Do NOT install transformers-4.40.0 from pip. The local `transformers/` directory contains necessary modifications for early-exit layer support.

4. **Download required models**

- **LLaMA-2-7B-Chat**: Download from [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- **Safety Classifier**: [zjunlp/SafeEdit-Safety-Classifier](https://huggingface.co/zjunlp/SafeEdit-Safety-Classifier)

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
```

## Evaluation Metrics

The framework evaluates:

| Metric | Description |
|--------|-------------|
| **DS** | Defense Success rate |
| **DG_onlyQ** | Defense Generalization (question only) |
| **DG_otherA** | Defense Generalization (other adversarial prompts) |
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
@inproceedings{dong-etal-2025-dscd,
    title = "{DSCD}: Large Language Model Detoxification with Self-Constrained Decoding",
    author = "Dong, Ming  and
      Zhang, Jinkui  and
      Zheng, Bolong  and
      Tu, Xinhui  and
      Hu, Po  and
      He, Tingting",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.197/",
    doi = "10.18653/v1/2025.emnlp-main.197",
    pages = "3969--3984",
    ISBN = "979-8-89176-332-6",
    abstract = "Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work innovatively proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLMs detoxification without parameter fine-tuning. DSCD strengthens the inner token distribution of the safety layer while weakening that of hallucination and toxic layer during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD{'}s effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD{'}s potential as a practical and scalable solution for safer LLM deployments."
}
```

## Acknowledgments

This project builds upon:
- [EasyEdit](https://github.com/zjunlp/EasyEdit) - Knowledge editing framework
- [SafeEdit](https://huggingface.co/datasets/zjunlp/SafeEdit) - Safety editing dataset
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
