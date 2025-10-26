import pandas as pd
import json
from sklearn.model_selection import train_test_split

def create_instruction_tuning_dataset(csv_path, train_output_path, val_output_path, test_size=0.1):
    """
    Converts a CSV of employee data into a JSONL dataset for instruction fine-tuning.

    Args:
        csv_path (str): Path to the input CSV file.
        train_output_path (str): Path to save the training JSONL file.
        val_output_path (str): Path to save the validation JSONL file.
        test_size (float): Proportion of the dataset to include in the validation split.
    """
    df = pd.read_csv(csv_path)

    def format_employee_profile(row):
        return (
            f"Employee Profile:\n"
            f"- Hire Date: {row['hire_date']}\n"
            f"- Title: {row['title']}\n"
            f"- Department: {row['department']}\n"
            f"- Region: {row['region']}\n"
            f"- Last Promotion: {row.get('promotion_date', 'N/A')}\n"
            f"- Performance: {row['performance_rating']}"
        )

    def format_assistant_response(row):
        tenure = row['actual_tenure']
        promotion_time = row.get('promotion_time', 'N/A')
        attrition = 'High' if row['attrition'] else 'Low'

        return (
            f"Based on this profile:\n\n"
            f"1. Expected Tenure: {tenure} years\n"
            f"   Reasoning: [Model should generate reasoning based on patterns in the data]\n\n"
            f"2. Time to Next Promotion: {promotion_time} years\n"
            f"   Reasoning: [Model should generate reasoning based on patterns in the data]\n\n"
            f"3. Attrition Risk: {attrition}\n"
            f"   Reasoning: [Model should generate reasoning based on patterns in the data]"
        )

    def create_jsonl_entry(row):
        user_content = (
            f"{format_employee_profile(row)}\n\n"
            f"Predict: 1) Expected tenure in current role, 2) Time to next promotion, 3) Attrition risk"
        )
        
        assistant_content = format_assistant_response(row)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an HR analytics expert. Predict employee tenure, promotion time, and attrition risk based on employee profiles."
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    with open(train_output_path, 'w') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(create_jsonl_entry(row)) + '\n')

    with open(val_output_path, 'w') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(create_jsonl_entry(row)) + '\n')

    print(f"Successfully created {train_output_path} and {val_output_path}")

if __name__ == "__main__":
    # Create a dummy CSV for demonstration purposes
    data = {
        'hire_date': ['2020-01-15', '2021-03-10', '2019-11-20'],
        'promotion_date': ['2022-03-20', None, '2021-05-10'],
        'title': ['Senior Engineer', 'Marketing Manager', 'Senior Analyst'],
        'department': ['Engineering', 'Marketing', 'Finance'],
        'region': ['North America', 'East Coast', 'Midwest'],
        'performance_rating': ['High', 'Meets Expectations', 'High'],
        'actual_tenure': [3.2, 1.8, 4.5],
        'promotion_time': [2.1, None, 1.8],
        'attrition': [False, True, False]
    }
    dummy_df = pd.DataFrame(data)
    dummy_csv_path = 'dummy_employee_data.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)

    create_instruction_tuning_dataset(
        csv_path=dummy_csv_path,
        train_output_path='train_dataset.jsonl',
        val_output_path='val_dataset.jsonl'
    )

