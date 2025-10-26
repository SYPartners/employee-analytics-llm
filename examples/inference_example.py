"""
Example script for running inference with the fine-tuned employee analytics model.

This demonstrates both population-level and personalized predictions.
"""

from transformers import pipeline
import json

def load_model(model_path="./employee_analytics_model"):
    """Load the fine-tuned model."""
    pipe = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return pipe

def format_employee_profile(employee_data):
    """Format employee data as a text profile."""
    profile = "Employee Profile:\n"
    for key, value in employee_data.items():
        profile += f"- {key}: {value}\n"
    return profile

def get_population_prediction(pipe, employee_profile):
    """Get population-level prediction for an employee."""
    prompt = f"""{employee_profile}
Predict: 1) Expected tenure in current role, 2) Time to next promotion, 3) Attrition risk"""
    
    messages = [
        {"role": "system", "content": "You are an HR analytics expert. Predict employee tenure, promotion time, and attrition risk based on employee profiles."},
        {"role": "user", "content": prompt}
    ]
    
    outputs = pipe(messages, max_new_tokens=512)
    return outputs[0]["generated_text"][-1]["content"]

def get_personalized_prediction(pipe, base_prediction, individual_history):
    """Get personalized prediction based on individual history."""
    personalization_prompt = f"""
The population-level prediction for this employee is:
{base_prediction}

Now, consider this individual's specific history:
{individual_history}

Adjust the prediction based on this new information. Provide a personalized prediction with updated reasoning.
"""
    
    messages = [
        {"role": "system", "content": "You are an HR analytics expert."},
        {"role": "user", "content": personalization_prompt}
    ]
    
    outputs = pipe(messages, max_new_tokens=512)
    return outputs[0]["generated_text"][-1]["content"]

def main():
    # Load the model
    print("Loading model...")
    pipe = load_model()
    
    # Example employee data
    employee_data = {
        "Hire Date": "2023-01-10",
        "Title": "Software Engineer",
        "Department": "Technology",
        "Region": "West Coast",
        "Last Promotion": "N/A",
        "Performance": "Meets Expectations"
    }
    
    # Format profile
    employee_profile = format_employee_profile(employee_data)
    
    # Get population-level prediction
    print("\n" + "="*60)
    print("POPULATION-LEVEL PREDICTION")
    print("="*60)
    base_prediction = get_population_prediction(pipe, employee_profile)
    print(base_prediction)
    
    # Individual history for personalization
    individual_history = """
- Actual tenure so far: 1.5 years
- Project completion rate: 95%
- Manager feedback: Exceeds expectations in technical skills, needs development in leadership
- Engagement survey score: 85/100
- Recent achievements: Led successful migration project, mentored 2 junior engineers
"""
    
    # Get personalized prediction
    print("\n" + "="*60)
    print("PERSONALIZED PREDICTION")
    print("="*60)
    personalized_prediction = get_personalized_prediction(
        pipe, 
        base_prediction, 
        individual_history
    )
    print(personalized_prediction)

if __name__ == "__main__":
    main()

