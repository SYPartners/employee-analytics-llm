# OpenAI GPT-OSS-120B Model Research Notes

## Overview
OpenAI released gpt-oss-120b and gpt-oss-20b in August 2025 as their first open-weight models since GPT-2.

## Model Specifications

### GPT-OSS-120B
- **Total Parameters**: 117 billion (referred to as 120B)
- **Active Parameters**: 5.1 billion (Mixture of Experts architecture)
- **Architecture**: Mixture of Experts (MoE) with MXFP4 quantization
- **Memory Requirements**: Fits in a single 80GB GPU (NVIDIA H100 or AMD MI300X)
- **License**: Apache 2.0 (permissive, commercial use allowed)

### GPT-OSS-20B
- **Total Parameters**: 21 billion
- **Active Parameters**: 3.6 billion
- **Memory Requirements**: 16GB of memory
- **Use Case**: Lower latency, local or specialized applications

## Key Features

### 1. Mixture of Experts (MoE) Architecture
- Only 5.1B parameters are active during inference (out of 117B total)
- Enables running a large model on consumer hardware
- MXFP4 quantization of MoE weights reduces memory footprint

### 2. Harmony Response Format
- Models were trained on OpenAI's "harmony response format"
- **Critical**: Models MUST be used with harmony format or they won't work correctly
- Format is automatically applied when using Transformers chat template

### 3. Configurable Reasoning Effort
- **Low**: Fast responses for general dialogue
- **Medium**: Balanced speed and detail
- **High**: Deep and detailed analysis
- Set via system prompt: "Reasoning: high"

### 4. Full Chain-of-Thought Access
- Complete access to model's reasoning process
- Facilitates debugging and increases trust
- Not intended for end users

### 5. Agentic Capabilities
- Function calling with defined schemas
- Web browsing (built-in tools)
- Python code execution
- Structured Outputs
- Designed for agentic operations

### 6. Fine-tunable
- Can be fine-tuned on a single H100 node
- Fully customizable for specific use cases
- Parameter fine-tuning supported

## Hardware Requirements

### Single DGX Spark (128 GB)
- **Can run**: gpt-oss-120b with MXFP4 quantization (requires ~80GB)
- **Remaining memory**: ~48GB for batch processing and context
- **Suitable for**: Inference, small-scale fine-tuning

### Two DGX Sparks (256 GB combined)
- **Can run**: Full fine-tuning of gpt-oss-120b
- **Distributed training**: Possible with PyTorch DDP
- **Larger batch sizes**: More efficient training

## Use Cases

### Designed For:
1. **Reasoning tasks**: Complex problem-solving with adjustable reasoning depth
2. **Agentic applications**: Function calling, tool use, web browsing
3. **Production deployments**: Apache 2.0 license allows commercial use
4. **Custom fine-tuning**: Adapt to domain-specific tasks

### NOT Designed For:
- Traditional tabular data prediction
- Time-series forecasting
- Structured data regression (without significant adaptation)

## Deployment Options

### Inference Frameworks
- **Transformers**: Native HuggingFace support
- **vLLM**: OpenAI-compatible server
- **Ollama**: Consumer hardware deployment
- **LM Studio**: Desktop application
- **PyTorch/Triton**: Custom implementations

### Cloud Providers
- AWS
- Oracle Cloud
- Multiple inference providers (Groq, Fireworks, Together AI, etc.)

## Technical Details

### Model Architecture
- Transformer-based with MoE layers
- BF16 weights for most layers
- MXFP4 quantization for MoE projection weights
- Optimized for single-GPU inference

### Training
- Trained on harmony response format
- Post-training with MXFP4 quantization
- All evaluations performed with MXFP4 quantization

## Citation
```
@misc{openai2025gptoss120bgptoss20bmodel,
      title={gpt-oss-120b & gpt-oss-20b Model Card}, 
      author={OpenAI},
      year={2025},
      eprint={2508.10925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10925}, 
}
```

## Key Considerations for Employee Analytics

### Challenges:
1. **Not designed for structured data**: GPT-OSS is a language model, not a tabular data model
2. **Requires text-based prompting**: Would need to convert employee data to text format
3. **Overkill for simple predictions**: Traditional ML models are more efficient for regression tasks
4. **Fine-tuning complexity**: Requires significant data and compute to adapt to structured prediction

### Potential Approach:
1. **Text-based feature engineering**: Convert employee records to natural language descriptions
2. **Few-shot learning**: Provide examples of employee scenarios and outcomes
3. **Chain-of-thought reasoning**: Leverage model's reasoning capabilities for complex predictions
4. **Hybrid approach**: Use LLM for feature extraction, traditional ML for final predictions

