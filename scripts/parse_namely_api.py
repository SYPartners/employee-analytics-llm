"""
Parse the Namely API response and convert to CSV format.
"""

import json
import pandas as pd

def parse_namely_api(json_path, output_csv_path):
    """Parse Namely API JSON response and convert to CSV."""
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract report data
    report = data['reports'][0]
    column_defs = report['columns']
    content = report['content']
    
    # Extract column labels
    columns = [col['label'] for col in column_defs]
    
    print(f"Columns ({len(columns)}):")
    for i, col in enumerate(column_defs):
        print(f"  {i}: {col['label']}")
    
    print(f"\nTotal records: {len(content)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(content, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Saved to {output_csv_path}")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    # Count unique employees
    unique_employees = df[['Last name', 'First name']].drop_duplicates()
    print(f"  Unique employees: {len(unique_employees)}")
    
    # Check for key columns
    if 'User status' in df.columns:
        print(f"\nEmployee Status:")
        print(df['User status'].value_counts())
    
    if 'Departments' in df.columns:
        print(f"\nTop Departments:")
        print(df['Departments'].value_counts().head(10))
    
    return df

if __name__ == "__main__":
    df = parse_namely_api(
        'data/namely_api_response.json',
        'data/namely_full_dataset.csv'
    )

