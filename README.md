# PRIME-VQA: Progressive Reasoning through Integrated Multimodal Enhancement

Official implementation of "Progressive Reasoning through Integrated Multimodal Enhancement for Knowledge-Driven Medical Visual Question Answering"

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](link-to-arxiv)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

PRIME-VQA is a knowledge-driven Medical Visual Question Answering framework that addresses training inefficiency and shallow multimodal integration through:
- **Adaptive Curriculum Learning**: Progressive difficulty ordering based on question complexity, answer depth, and medical terminology density
- **Cross-Modal Transformer Fusion**: Explicit integration of reference images, UMLS knowledge, and intra-class visual prototypes

### Key Results
- **Slake**: 87.95% accuracy (+2.15% over baselines)
- **VQA-RAD**: 77.4% accuracy (+0.7% over baselines)
- **Efficiency**: 3.1B parameters, 26.1 A100-hours training

## Architecture

![Architecture](assets/architecture.png)

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/PRIME-VQA.git
cd PRIME-VQA

# Create conda environment
conda create -n primevqa python=3.8
conda activate primevqa

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Data Preparation

### Download Datasets

```bash
# Download Slake dataset
python scripts/download_data.py --dataset slake --output_dir data/slake

# Download VQA-RAD dataset
python scripts/download_data.py --dataset vqarad --output_dir data/vqarad
```

### Knowledge Source Construction

```bash
# Extract medical entities and build UMLS knowledge base
python scripts/build_knowledge_source.py \
    --dataset slake \
    --data_dir data/slake \
    --output_dir data/knowledge/slake \
    --umls_path /path/to/umls
```

## Training

### 1. Knowledge Space Pretraining

```bash
python train_pretraining.py \
    --config configs/pretraining/slake.yaml \
    --knowledge_dir data/knowledge/slake \
    --output_dir checkpoints/pretraining/slake \
    --num_epochs 300 \
    --batch_size 48 \
    --gpu 0
```

### 2. Curriculum-Based VQA Training

```bash
python train_vqa.py \
    --config configs/vqa/slake_llama3b.yaml \
    --data_dir data/slake \
    --knowledge_dir data/knowledge/slake \
    --pretrained_ckpt checkpoints/pretraining/slake/best_model.pth \
    --output_dir checkpoints/vqa/slake_llama3b \
    --curriculum \
    --num_epochs 10 \
    --batch_size 8 \
    --gpu 0
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 train_vqa.py \
    --config configs/vqa/slake_llama3b.yaml \
    --distributed
```

## Evaluation

```bash
python evaluate.py \
    --config configs/vqa/slake_llama3b.yaml \
    --checkpoint checkpoints/vqa/slake_llama3b/best_model.pth \
    --data_dir data/slake \
    --split test \
    --output_dir results/slake
```

## Inference

### Single Sample Inference

```python
from primevqa import PRIMEVQA
from PIL import Image

# Load model
model = PRIMEVQA.from_pretrained("checkpoints/vqa/slake_llama3b/best_model.pth")

# Load image and question
image = Image.open("path/to/medical_image.jpg")
question = "What does the blunting of the costophrenic angles indicate?"

# Generate answer
answer = model.generate(image, question)
print(f"Answer: {answer}")
```

### Batch Inference

```bash
python inference.py \
    --checkpoint checkpoints/vqa/slake_llama3b/best_model.pth \
    --input_file data/test_samples.json \
    --output_file results/predictions.json \
    --batch_size 16
```

## Visualization

### Attention Visualization

```python
from primevqa.visualization import visualize_attention

# Visualize cross-modal attention
visualize_attention(
    model=model,
    image=image,
    question=question,
    output_dir="visualizations/attention"
)
```

### Difficulty Distribution

```bash
python scripts/visualize_curriculum.py \
    --data_dir data/slake \
    --output_dir visualizations/difficulty
```

## Model Zoo

| Model | Dataset | Accuracy | Checkpoint |
|-------|---------|----------|------------|
| PRIME-VQA (Llama-3B) | Slake | 87.95% | [Download](link) |
| PRIME-VQA (Llama-3B) | VQA-RAD | 77.4% | [Download](link) |
| PRIME-VQA (GPT2-XL) | Slake | 85.4% | [Download](link) |

## Repository Structure

```
PRIME-VQA/
├── configs/                    # Configuration files
│   ├── pretraining/
│   │   ├── slake.yaml
│   │   └── vqarad.yaml
│   └── vqa/
│       ├── slake_llama3b.yaml
│       └── vqarad_llama3b.yaml
├── data/                       # Data directory (gitignored)
│   ├── slake/
│   ├── vqarad/
│   └── knowledge/
├── primevqa/                   # Main package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py        # BiomedCLIP, GLIMS, PubMedBERT
│   │   ├── fusion.py          # Cross-modal transformer
│   │   ├── curriculum.py      # Curriculum learning
│   │   └── prime_vqa.py       # Main model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── knowledge.py       # Knowledge source construction
│   │   └── transforms.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py          # BaMCo, CE losses
│   │   └── optimizer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── checkpoint.py
│   └── visualization/
│       ├── __init__.py
│       ├── attention.py
│       └── difficulty.py
├── scripts/
│   ├── download_data.py
│   ├── build_knowledge_source.py
│   ├── visualize_curriculum.py
│   └── export_model.py
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
├── assets/
│   └── architecture.png
├── train_pretraining.py
├── train_vqa.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## Configuration

Example configuration file (`configs/vqa/slake_llama3b.yaml`):

```yaml
# Model Configuration
model:
  llm:
    name: "llama-3.2-3b"
    lora_r: 16
    lora_alpha: 32
  
  encoders:
    image: "biomedclip-vit-b16"  # frozen
    knowledge: "pubmedbert-base"  # frozen
    intraclass: "glims"           # frozen
  
  fusion:
    num_layers: 4
    d_model: 3072
    num_heads: 8
    d_ff: 12288
    dropout: 0.15

# Training Configuration
training:
  num_epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  weight_decay: 1e-4
  warmup_steps: 100
  
  curriculum:
    enabled: true
    alpha: 0.5  # question length weight
    beta: 0.3   # answer length weight
    gamma: 1.0  # complexity weight
    delta: 1.5  # term density weight
    pacing_start: 0.3
    pacing_end: 1.0

# Data Configuration
data:
  train_split: "train"
  val_split: "val"
  test_split: "test"
  num_workers: 4
  intraclass_samples: 36

# Knowledge Configuration
knowledge:
  umls_path: "/path/to/umls"
  max_text_length: 256
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{primevqa2025,
  title={Progressive Reasoning through Integrated Multimodal Enhancement for Knowledge-Driven Medical Visual Question Answering},
  author={Your Name and Others},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BiomedCLIP](https://github.com/microsoft/BiomedCLIP) for the pretrained vision encoder
- [GLIMS](https://github.com/link) for the multi-scale image encoder
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) for the text encoder
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B) for the language model backbone

## Contact

For questions or issues, please:
- Open an issue on GitHub

---

**Note**: This is a research project. Please ensure compliance with data usage agreements and ethical guidelines when using medical datasets.
