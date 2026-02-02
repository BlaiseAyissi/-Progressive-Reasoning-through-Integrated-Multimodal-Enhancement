# PRIME-VQA: Progressive Reasoning through Integrated Multimodal Enhancement

Official implementation of "Progressive Reasoning through Integrated Multimodal Enhancement for Knowledge-Driven Medical Visual Question Answering"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

PRIME-VQA is a knowledge-driven Medical Visual Question Answering framework that addresses training inefficiency and shallow multimodal integration through:
- **Adaptive Curriculum Learning**: Progressive difficulty ordering based on question complexity, answer depth, and medical terminology density
- **Cross-Modal Transformer Fusion**: Explicit integration of reference images, UMLS knowledge, and intra-class visual prototypes

### Key Results
- **Slake**: 87.95% accuracy (+2.15% over baselines)
- **VQA-RAD**: 77.4% accuracy (+0.7% over baselines)
- **Efficiency**: 3.1B parameters, 26.1 A100-hours training


## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/BlaiseAyissi/-Progressive-Reasoning-through-Integrated-Multimodal-Enhancement.git
cd PRIME-VQA

# Create conda environment
conda create -n primevqa python=3.8
conda activate primevqa

# Install dependencies
pip install -r requirements.txt

## Model Zoo

| Model | Dataset | Accuracy | Checkpoint |
|-------|---------|----------|------------|
| PRIME-VQA (Llama-3B) | Slake | 87.95% | 
| PRIME-VQA (Llama-3B) | VQA-RAD | 77.4% | 
| PRIME-VQA (GPT2-XL) | Slake | 85.4% |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub

---

**Note**: This is a research project. Please ensure compliance with data usage agreements and ethical guidelines when using medical datasets.
