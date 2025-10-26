# Research: Using LLMs for Employee Attrition Prediction

## Key Paper: "Can Large Language Model Predict Employee Attrition?"
**Authors**: Xiaoye Ma, Weiheng Liu, Changyi Zhao, Liliya R. Tukhvatulina  
**Institution**: Tomsk State University, Tomsk, Russia  
**Published**: November 2024  
**arXiv**: 2411.01353  

## Executive Summary

This groundbreaking study demonstrates that **fine-tuned GPT-3.5 significantly outperforms traditional machine learning models** for employee attrition prediction, achieving an F1-score of 0.92 compared to 0.82 for the best traditional model (SVM).

## Methodology

### Dataset
- **Source**: IBM HR Analytics Employee Attrition dataset
- **Features**: Employee demographics, job characteristics, performance metrics, work-life balance indicators
- **Task**: Binary classification (attrition vs. retention)

### Models Compared

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Fine-tuned GPT-3.5** | **0.91** | **0.94** | **0.92** |
| Support Vector Machine (SVM) | 0.81 | 0.83 | 0.82 |
| Random Forest | 0.79 | 0.81 | 0.80 |
| XGBoost | 0.78 | 0.82 | 0.80 |
| AdaBoost | 0.77 | 0.81 | 0.79 |
| Logistic Regression | 0.76 | 0.80 | 0.78 |
| K-Nearest Neighbors (KNN) | 0.69 | 0.73 | 0.71 |

### Key Findings

1. **GPT-3.5 outperformed all traditional ML models by 10+ percentage points** in F1-score
2. **Enhanced interpretability**: LLM identified subtle linguistic cues and recurring themes in employee behavior
3. **Complex pattern recognition**: LLM captured nuanced relationships that traditional models missed
4. **Contextual understanding**: Ability to interpret employee communications and detect subtle attrition signals

## Why LLMs Excel at Employee Attrition

### 1. Contextual Understanding
- Traditional ML treats features as independent variables
- LLMs understand relationships and context between features
- Can interpret semantic meaning of job titles, departments, feedback

### 2. Pattern Recognition
- Identifies subtle behavioral patterns across multiple dimensions
- Captures non-linear relationships in employee data
- Learns from textual descriptions and qualitative data

### 3. Interpretability
- Provides reasoning for predictions via chain-of-thought
- Identifies specific factors contributing to attrition risk
- Generates human-readable explanations

## Application to Your Use Case

Your requirements:
1. Predict average tenure in role
2. Predict attrition risk
3. Predict average promotion time
4. Provide both population-level and personalized predictions

### How GPT-OSS-120B Can Address This

#### Approach 1: Fine-Tuning (Recommended)
Based on the research paper's success with GPT-3.5, fine-tuning GPT-OSS-120B on your employee data would likely yield excellent results.

**Data Preparation**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an HR analytics expert. Predict employee tenure, promotion time, and attrition risk based on employee profiles."
    },
    {
      "role": "user",
      "content": "Employee Profile:\n- Hire Date: 2020-01-15\n- Title: Senior Engineer\n- Department: Engineering\n- Region: North America\n- Last Promotion: 2022-03-20\n- Performance: High\n\nPredict: 1) Expected tenure in current role, 2) Time to next promotion, 3) Attrition risk"
    },
    {
      "role": "assistant",
      "content": "Based on this profile:\n\n1. Expected Tenure: 2.8 years\n   Reasoning: High performer in Engineering with recent promotion shows career progression. Typical tenure for this level is 2-3 years before next move.\n\n2. Time to Next Promotion: 1.5 years\n   Reasoning: 2-year gap between last promotion suggests 18-month promotion cycle. High performance accelerates timeline.\n\n3. Attrition Risk: Low (15%)\n   Reasoning: Recent promotion, high performance, and career progression indicate strong engagement. Engineering retention is typically high for promoted employees."
    }
  ]
}
```

**Fine-Tuning Process**:
1. Convert historical employee data to instruction-following format
2. Include examples with known outcomes (actual tenure, promotions, attrition)
3. Fine-tune GPT-OSS-120B using your two DGX Sparks with distributed training
4. Validate on held-out test set

#### Approach 2: Few-Shot Learning (No Fine-Tuning Required)
For immediate deployment without fine-tuning:

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="openai/gpt-oss-120b",
    torch_dtype="auto",
    device_map="auto",
)

# Few-shot prompt with examples
prompt = """You are an HR analytics expert specializing in employee retention and career progression.

Example 1:
Employee: Hired 2018-06-01, Software Engineer, Engineering, West Coast, Promoted 2020-08-15
Predictions: Tenure: 3.2 years, Promotion Time: 2.1 years, Attrition Risk: Low (12%)

Example 2:
Employee: Hired 2021-03-10, Marketing Manager, Marketing, East Coast, No promotions
Predictions: Tenure: 1.8 years, Promotion Time: N/A, Attrition Risk: High (68%)

Example 3:
Employee: Hired 2019-11-20, Senior Analyst, Finance, Midwest, Promoted 2021-05-10, 2023-09-15
Predictions: Tenure: 4.5 years, Promotion Time: 1.8 years average, Attrition Risk: Very Low (5%)

Now predict for this employee:
Hired: 2022-01-15
Title: Data Scientist
Department: Product
Region: Remote
Last Promotion: 2023-07-20
Performance Rating: Exceeds Expectations

Provide detailed predictions with reasoning."""

outputs = pipe(prompt, max_new_tokens=512)
print(outputs[0]["generated_text"])
```

#### Approach 3: Personalization Layer
For personalized predictions based on individual history:

```python
# Base prediction from population model
base_prediction = get_population_prediction(employee_profile)

# Personalization prompt
personalization_prompt = f"""
Population-level prediction for this employee profile:
- Expected Tenure: {base_prediction['tenure']} years
- Promotion Time: {base_prediction['promotion_time']} years  
- Attrition Risk: {base_prediction['attrition_risk']}%

Individual history for this specific employee:
- Actual tenure so far: {employee['current_tenure']} years
- Promotion history: {employee['promotion_dates']}
- Performance trajectory: {employee['performance_history']}
- Engagement scores: {employee['engagement_scores']}

Adjust the population prediction based on this individual's actual history and trajectory. 
Provide personalized predictions that account for their specific patterns.
"""

personalized_output = pipe(personalization_prompt, max_new_tokens=512)
```

## Implementation on Two DGX Sparks

### Memory Allocation
- **DGX Spark 1**: Load GPT-OSS-120B model (~80GB), leaving ~48GB for data
- **DGX Spark 2**: Replica for distributed fine-tuning or inference scaling

### Fine-Tuning Setup

```bash
# On DGX Spark 1 (Master)
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<SPARK_1_IP> \
    --master_port=12355 \
    finetune_gpt_oss.py \
    --model_name openai/gpt-oss-120b \
    --train_file employee_data_train.jsonl \
    --val_file employee_data_val.jsonl \
    --output_dir ./finetuned_model \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4

# On DGX Spark 2 (Worker)
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<SPARK_1_IP> \
    --master_port=12355 \
    finetune_gpt_oss.py \
    [same arguments as above]
```

### Inference Deployment

```python
# Load fine-tuned model on single DGX Spark
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./finetuned_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

# Batch inference for multiple employees
def predict_batch(employee_profiles):
    prompts = [format_employee_prompt(emp) for emp in employee_profiles]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return [tokenizer.decode(out) for out in outputs]
```

## Expected Performance

Based on the research paper's findings:
- **Attrition Prediction**: F1-score of 0.90+ (vs 0.80 for traditional ML)
- **Interpretability**: Clear reasoning for each prediction
- **Flexibility**: Easy to adapt to new features or questions
- **Personalization**: Strong capability to adjust based on individual history

## Advantages Over Traditional ML

1. **Higher Accuracy**: 10-12% improvement in F1-score
2. **Interpretability**: Natural language explanations for predictions
3. **Flexibility**: No need to retrain for new questions or features
4. **Multi-Task**: Single model handles tenure, promotion, and attrition
5. **Contextual**: Understands relationships between features
6. **Personalization**: Easy to incorporate individual history

## Challenges

1. **Computational Cost**: Requires significant GPU resources (addressed by DGX Sparks)
2. **Latency**: Slower than traditional ML (seconds vs milliseconds)
3. **Data Requirements**: Fine-tuning needs substantial labeled data
4. **Consistency**: May produce slightly different outputs on repeated runs
5. **Cost**: Fine-tuning and inference have higher compute costs

## Recommendation

**Proceed with GPT-OSS-120B Fine-Tuning Approach**

Given:
- Research evidence showing 10+ point F1-score improvement
- Your access to two DGX Sparks (sufficient for fine-tuning)
- Need for interpretability in HR decisions
- Requirement for both population and personalized predictions

**Implementation Plan**:
1. **Phase 1** (Week 1-2): Prepare employee data in instruction-following format
2. **Phase 2** (Week 3-4): Fine-tune GPT-OSS-120B on two DGX Sparks
3. **Phase 3** (Week 5): Validate model performance vs traditional ML baseline
4. **Phase 4** (Week 6): Deploy inference pipeline and build personalization layer
5. **Phase 5** (Week 7+): Monitor performance and iterate

## References

1. Ma, X., Liu, W., Zhao, C., & Tukhvatulina, L. R. (2024). Can Large Language Model Predict Employee Attrition? arXiv:2411.01353. https://arxiv.org/abs/2411.01353

2. OpenAI. (2025). GPT-OSS-120B Model Card. https://huggingface.co/openai/gpt-oss-120b

3. Fang, X. et al. (2024). Large Language Models on Tabular Data: A Survey. arXiv:2402.17944.

