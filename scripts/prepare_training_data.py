"""
Prepare training data for fine-tuning GPT-OSS-120B.

This script converts the processed employee data into instruction-following format
suitable for fine-tuning large language models.
"""

import json
import random
from sklearn.model_selection import train_test_split

def format_employee_profile(employee):
    """Format employee data as a text profile."""
    profile = f"""Employee Profile:
- Hire Date: {employee['hire_date']}
- Department: {employee['current_department']}
- Level: {employee['current_level']}
- Step: {employee['current_step']}
- Title: {employee['current_title']}
- Location: {employee['current_location']}
- Division: {employee['current_division']}
- Number of Promotions: {employee['num_promotions']}
- Time in Current Role: {employee['time_in_current_role_years']} years"""
    
    return profile

def format_prediction_output(employee):
    """Format the expected prediction output."""
    
    # Determine attrition risk level
    if employee['attrition']:
        attrition_risk = "High"
        attrition_reasoning = f"Employee has left the organization after {employee['tenure_years']} years of tenure."
    else:
        # Active employees - assess risk based on tenure and promotions
        if employee['tenure_years'] < 1:
            attrition_risk = "Medium"
            attrition_reasoning = "New employee (< 1 year tenure) - still in adjustment period."
        elif employee['num_promotions'] == 0 and employee['tenure_years'] > 3:
            attrition_risk = "Medium-High"
            attrition_reasoning = f"No promotions after {employee['tenure_years']} years may indicate stagnation."
        elif employee['num_promotions'] > 0:
            attrition_risk = "Low"
            attrition_reasoning = f"Employee has received {employee['num_promotions']} promotion(s) and shows career progression."
        else:
            attrition_risk = "Low-Medium"
            attrition_reasoning = f"Employee is progressing normally with {employee['tenure_years']} years tenure."
    
    # Format promotion time prediction
    if employee['avg_promotion_time_years']:
        promotion_time = f"{employee['avg_promotion_time_years']} years"
        promotion_reasoning = f"Based on historical pattern of {employee['num_promotions']} promotion(s) over {employee['tenure_years']} years."
    else:
        promotion_time = "No promotion history"
        promotion_reasoning = "Employee has not received promotions yet. Typical promotion time in organization is 2-3 years."
    
    # Format tenure prediction
    tenure_prediction = f"{employee['tenure_years']} years (actual)"
    tenure_reasoning = f"Employee joined on {employee['hire_date']} in {employee['current_department']} department."
    
    output = f"""Based on this employee profile:

1. **Tenure**: {tenure_prediction}
   *Reasoning*: {tenure_reasoning}

2. **Promotion Time**: {promotion_time}
   *Reasoning*: {promotion_reasoning}

3. **Attrition Risk**: {attrition_risk}
   *Reasoning*: {attrition_reasoning}"""
    
    return output

def create_training_example(employee):
    """Create a single training example in instruction-following format."""
    
    system_message = """You are an expert HR analytics consultant specializing in employee retention, career progression, and workforce planning. Your role is to analyze employee profiles and provide data-driven predictions about:

1. Expected tenure in current role
2. Time to next promotion
3. Attrition risk assessment

Provide clear reasoning for each prediction based on the employee's profile, historical patterns, and organizational context."""

    user_message = f"""{format_employee_profile(employee)}

Please analyze this employee profile and predict:
1. Expected tenure in current role
2. Time to next promotion  
3. Attrition risk (Low/Medium/High) with reasoning"""

    assistant_message = format_prediction_output(employee)
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

def prepare_training_data(input_json_path, output_train_path, output_val_path, test_size=0.15, random_seed=42):
    """
    Prepare training and validation datasets.
    
    Args:
        input_json_path: Path to processed employee JSON
        output_train_path: Path to save training JSONL
        output_val_path: Path to save validation JSONL
        test_size: Proportion for validation set
        random_seed: Random seed for reproducibility
    """
    
    print("="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    # Load processed employee data
    print(f"\nLoading employee data from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        employees = json.load(f)
    
    print(f"✓ Loaded {len(employees)} employee records")
    
    # Filter out employees with insufficient data
    valid_employees = [
        emp for emp in employees 
        if emp['current_department'] != 'Unknown' 
        and emp['tenure_years'] > 0
    ]
    
    print(f"✓ Filtered to {len(valid_employees)} employees with complete data")
    
    # Create training examples
    print("\nGenerating training examples...")
    training_examples = [create_training_example(emp) for emp in valid_employees]
    print(f"✓ Created {len(training_examples)} training examples")
    
    # Split into train and validation
    train_examples, val_examples = train_test_split(
        training_examples, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    print(f"\nSplit: {len(train_examples)} train, {len(val_examples)} validation")
    
    # Save training data
    print(f"\nSaving training data to {output_train_path}...")
    with open(output_train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Saved {len(train_examples)} training examples")
    
    # Save validation data
    print(f"\nSaving validation data to {output_val_path}...")
    with open(output_val_path, 'w') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Saved {len(val_examples)} validation examples")
    
    # Print sample
    print("\n" + "="*70)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*70)
    print("\nSystem Message:")
    print(train_examples[0]['messages'][0]['content'][:200] + "...")
    print("\nUser Message:")
    print(train_examples[0]['messages'][1]['content'][:300] + "...")
    print("\nAssistant Response:")
    print(train_examples[0]['messages'][2]['content'][:400] + "...")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    
    return train_examples, val_examples

if __name__ == "__main__":
    train_examples, val_examples = prepare_training_data(
        input_json_path='data/processed_employees_full.json',
        output_train_path='data/train_dataset.jsonl',
        output_val_path='data/val_dataset.jsonl',
        test_size=0.15,
        random_seed=42
    )

