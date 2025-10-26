"""
Two-stage inference script for employee analytics.

Stage 1: Population-level prediction.
Stage 2: Personalized prediction based on individual history.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# --- Configuration ---
MODEL_PATH = "./employee_analytics_model"

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    # Load model with appropriate quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response from the model based on conversation history."""
    
    # Format messages into a single string
    def format_messages(messages):
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|system|>\n{content}\n"
            elif role == 'user':
                formatted += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                formatted += f"<|assistant|>\n{content}\n"
        return formatted

    prompt = format_messages(messages)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the assistant's response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response

def get_population_prediction(model, tokenizer, employee_profile):
    """Stage 1: Get population-level prediction."""
    
    system_message = "You are an expert HR analytics consultant specializing in employee retention, career progression, and workforce planning."
    
    user_message = f"""{employee_profile}

Please analyze this employee profile and predict:
1. Expected tenure in current role
2. Time to next promotion  
3. Attrition risk (Low/Medium/High) with reasoning"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return generate_response(model, tokenizer, messages)

def get_personalized_prediction(model, tokenizer, base_prediction, individual_history):
    """Stage 2: Get personalized prediction based on individual history."""
    
    system_message = "You are an expert HR analytics consultant specializing in employee retention, career progression, and workforce planning."
    
    personalization_prompt = f"""
The population-level prediction for this employee is:
{base_prediction}

Now, consider this individual's specific history:
{individual_history}

Adjust the prediction based on this new information. Provide a personalized prediction with updated reasoning.
"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": personalization_prompt}
    ]
    
    return generate_response(model, tokenizer, messages)

def main():
    # 1. Load Model (Assume fine-tuning is complete and model is saved)
    # NOTE: This will fail if the model is not present.
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been fine-tuned and saved to the correct path.")
        return

    # 2. Example Employee Profile (New Hire Scenario)
    new_employee_profile = """Employee Profile:
- Hire Date: 2025-01-01
- Department: Program Management
- Level: Staff
- Step: Step 1
- Title: Project Coordinator
- Location: New York
- Division: Consulting
- Number of Promotions: 0
- Time in Current Role: 0.8 years"""
    
    # 3. Stage 1: Population-Level Prediction
    print("\n" + "="*60)
    print("STAGE 1: POPULATION-LEVEL PREDICTION (Based on organizational trends)")
    print("="*60)
    base_prediction = get_population_prediction(model, tokenizer, new_employee_profile)
    print(base_prediction)
    
    # 4. Example Individual History (for personalization)
    individual_history = """
- Actual tenure so far: 0.8 years
- Performance Review: Exceeds Expectations (Q3 2025)
- Manager Feedback: Ready for next level responsibilities, high engagement score (92/100)
- Training Completed: Advanced Project Management Certification
"""
    
    # 5. Stage 2: Personalized Prediction
    print("\n" + "="*60)
    print("STAGE 2: PERSONALIZED PREDICTION (Adjusted by individual history)")
    print("="*60)
    personalized_prediction = get_personalized_prediction(
        model, 
        tokenizer, 
        base_prediction, 
        individual_history
    )
    print(personalized_prediction)

if __name__ == "__main__":
    main()

