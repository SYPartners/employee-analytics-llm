# Employee Analytics with GPT-OSS-120B

A comprehensive solution for predicting employee tenure, attrition risk, and promotion time using OpenAI's open-source 120-billion-parameter language model (GPT-OSS-120B) on NVIDIA DGX Spark systems.

## Overview

This project leverages state-of-the-art Large Language Models (LLMs) to provide highly accurate and interpretable predictions for HR analytics. Recent research has shown that fine-tuned LLMs can achieve **92% F1-score** for employee attrition prediction, significantly outperforming traditional machine learning models (82% for SVM, 80% for Random Forest/XGBoost).

### Key Features

- **Multi-Task Prediction**: Single model predicts tenure, attrition risk, and promotion time
- **High Accuracy**: 10+ percentage point improvement over traditional ML approaches
- **Interpretability**: Natural language explanations for all predictions
- **Personalization**: Two-stage inference provides both population-level and personalized predictions
- **Distributed Training**: Optimized for dual DGX Spark deployment

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Employee Data (CSV)                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│         Data Preparation (Instruction Format)            │
│              scripts/prepare_training_data.py           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│          Fine-Tuning GPT-OSS-120B (Distributed)          │
│          DGX Spark 1 (Master) + DGX Spark 2 (Worker)     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  Two-Stage Inference                     │
│   Stage 1: Population Prediction | Stage 2: Personalized │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- 2x NVIDIA DGX Spark systems (128 GB unified memory each)
- Python 3.10+
- PyTorch 2.0+
- Transformers library
- Employee data in CSV format

### Installation

```bash
# Clone the repository
git clone https://github.com/SYPartners/employee-analytics-llm.git
cd employee-analytics-llm

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Convert your employee CSV data to instruction-following format:

```bash
python scripts/prepare_training_data.py \
    --input employee_data.csv \
    --train_output train_dataset.jsonl \
    --val_output val_dataset.jsonl
```

### Fine-Tuning

Launch distributed training across two DGX Spark systems:

**On DGX Spark 1 (Master):**
```bash
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<SPARK_1_IP> \
    --master_port=12355 \
    finetune.py \
    --model_name_or_path openai/gpt-oss-120b \
    --train_data_file train_dataset.jsonl \
    --validation_data_file val_dataset.jsonl \
    --output_dir ./employee_analytics_model
```

**On DGX Spark 2 (Worker):**
```bash
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<SPARK_1_IP> \
    --master_port=12355 \
    finetune.py \
    --model_name_or_path openai/gpt-oss-120b \
    --train_data_file train_dataset.jsonl \
    --validation_data_file val_dataset.jsonl \
    --output_dir ./employee_analytics_model
```

### Inference

Run predictions on new employee profiles:

```python
from transformers import pipeline

# Load fine-tuned model
pipe = pipeline(
    "text-generation",
    model="./employee_analytics_model",
    torch_dtype="auto",
    device_map="auto"
)

# Make prediction
employee_profile = """
Employee Profile:
- Hire Date: 2023-01-10
- Title: Software Engineer
- Department: Technology
- Region: West Coast
- Last Promotion: N/A
- Performance: Meets Expectations
"""

messages = [
    {"role": "system", "content": "You are an HR analytics expert."},
    {"role": "user", "content": f"{employee_profile}\n\nPredict: 1) Expected tenure, 2) Promotion time, 3) Attrition risk"}
]

output = pipe(messages, max_new_tokens=512)
print(output[0]["generated_text"][-1])
```

## Documentation

- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)**: Complete walkthrough from data prep to deployment
- **[Research Findings](docs/RESEARCH_FINDINGS.md)**: Academic research supporting this approach
- **[LLM Approaches](docs/LLM_APPROACHES.md)**: Comparison of different LLM strategies for structured data
- **[GPT-OSS-120B Specifications](docs/GPT_OSS_120B_SPECS.md)**: Technical details of the model

## Performance

Based on research findings ([Ma et al., 2024](https://arxiv.org/abs/2411.01353)):

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Fine-tuned GPT-3.5** | **0.91** | **0.94** | **0.92** |
| Support Vector Machine | 0.81 | 0.83 | 0.82 |
| Random Forest | 0.79 | 0.81 | 0.80 |
| XGBoost | 0.78 | 0.82 | 0.80 |
| Logistic Regression | 0.76 | 0.80 | 0.78 |

## Hardware Requirements

### Minimum (Inference Only)
- 1x DGX Spark (128 GB unified memory)
- ~80 GB for model weights (MXFP4 quantization)
- ~48 GB remaining for data and batch processing

### Recommended (Fine-Tuning)
- 2x DGX Spark (256 GB combined memory)
- 200 Gbps QSFP interconnect for distributed training
- 500 GB+ storage for model checkpoints

## Project Structure

```
employee-analytics-llm/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation
│   ├── IMPLEMENTATION_GUIDE.md  # Complete implementation walkthrough
│   ├── RESEARCH_FINDINGS.md     # Academic research and findings
│   ├── LLM_APPROACHES.md        # Different LLM strategies
│   └── GPT_OSS_120B_SPECS.md    # Model specifications
├── scripts/                     # Utility scripts
│   └── prepare_training_data.py # Data preparation script
└── examples/                    # Example code and notebooks
    └── (coming soon)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## References

1. Ma, X., Liu, W., Zhao, C., & Tukhvatulina, L. R. (2024). *Can Large Language Model Predict Employee Attrition?* arXiv:2411.01353. [https://arxiv.org/abs/2411.01353](https://arxiv.org/abs/2411.01353)

2. OpenAI. (2025). *GPT-OSS-120B Model Card*. [https://huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)

3. Fang, X. et al. (2024). *Large Language Models on Tabular Data: A Survey*. arXiv:2402.17944.

## Acknowledgments

- OpenAI for releasing GPT-OSS-120B under Apache 2.0 license
- NVIDIA for DGX Spark hardware specifications and support
- Research community for advancing LLM applications in HR analytics

## Contact

For questions or support, please open an issue on GitHub.

