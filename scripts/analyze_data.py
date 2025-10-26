"""
Analyze the salary history data and prepare it for model training.

This script:
1. Loads and analyzes the salary history CSV
2. Calculates key metrics (tenure, promotion frequency, attrition)
3. Creates derived features for training
4. Generates summary statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_and_clean_data(csv_path):
    """Load the salary history CSV and perform initial cleaning."""
    df = pd.read_csv(csv_path)
    
    # Convert date columns to datetime
    df['Salary Start Date'] = pd.to_datetime(df['Salary Start Date'], format='%m/%d/%Y')
    df['Salary End Date'] = pd.to_datetime(df['Salary End Date'], format='%m/%d/%Y', errors='coerce')
    df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y')
    
    # Convert salary to numeric
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    
    # Sort by employee and salary start date
    df = df.sort_values(['Employee Number', 'Salary Start Date'])
    
    return df

def calculate_employee_metrics(df):
    """Calculate key metrics for each employee."""
    employees = {}
    
    for emp_num, emp_data in df.groupby('Employee Number'):
        emp_data = emp_data.sort_values('Salary Start Date')
        
        # Basic info
        first_name = emp_data.iloc[0]['First name']
        last_name = emp_data.iloc[0]['Last name']
        hire_date = emp_data.iloc[0]['Start date']
        current_status = emp_data.iloc[-1]['User status']
        
        # Calculate tenure
        if current_status == 'Active Employee':
            tenure_days = (datetime.now() - hire_date).days
            attrition = False
        else:
            # Find last salary end date or use last salary start date
            last_date = emp_data.iloc[-1]['Salary End Date']
            if pd.isna(last_date):
                last_date = emp_data.iloc[-1]['Salary Start Date']
            tenure_days = (last_date - hire_date).days
            attrition = True
        
        tenure_years = tenure_days / 365.25
        
        # Count promotions (salary increases > 10%)
        promotions = []
        for i in range(1, len(emp_data)):
            prev_salary = emp_data.iloc[i-1]['Salary']
            curr_salary = emp_data.iloc[i]['Salary']
            if pd.notna(prev_salary) and pd.notna(curr_salary):
                increase_pct = ((curr_salary - prev_salary) / prev_salary) * 100
                if increase_pct > 10:
                    promotions.append({
                        'date': emp_data.iloc[i]['Salary Start Date'],
                        'increase_pct': increase_pct,
                        'notes': emp_data.iloc[i]['Salary Notes']
                    })
        
        # Calculate average time between promotions
        if len(promotions) > 1:
            promotion_intervals = []
            for i in range(1, len(promotions)):
                days_between = (promotions[i]['date'] - promotions[i-1]['date']).days
                promotion_intervals.append(days_between / 365.25)
            avg_promotion_time = np.mean(promotion_intervals)
        elif len(promotions) == 1:
            # Time from hire to first promotion
            avg_promotion_time = (promotions[0]['date'] - hire_date).days / 365.25
        else:
            avg_promotion_time = None
        
        # Current role information
        current_dept = emp_data.iloc[-1]['Departments']
        current_level = emp_data.iloc[-1]['Level']
        current_step = emp_data.iloc[-1]['Step']
        current_title = emp_data.iloc[-1]['Job Title']
        current_location = emp_data.iloc[-1]['Office Locations']
        
        employees[emp_num] = {
            'employee_number': emp_num,
            'first_name': first_name,
            'last_name': last_name,
            'hire_date': hire_date,
            'tenure_years': tenure_years,
            'attrition': attrition,
            'status': current_status,
            'num_promotions': len(promotions),
            'avg_promotion_time_years': avg_promotion_time,
            'current_department': current_dept,
            'current_level': current_level,
            'current_step': current_step,
            'current_title': current_title,
            'current_location': current_location,
            'promotion_history': promotions,
            'salary_history_count': len(emp_data)
        }
    
    return employees

def generate_summary_statistics(employees):
    """Generate summary statistics from employee data."""
    stats = {
        'total_employees': len(employees),
        'active_employees': sum(1 for e in employees.values() if e['status'] == 'Active Employee'),
        'inactive_employees': sum(1 for e in employees.values() if e['status'] != 'Active Employee'),
        'attrition_rate': sum(1 for e in employees.values() if e['attrition']) / len(employees) * 100,
    }
    
    # Tenure statistics
    tenures = [e['tenure_years'] for e in employees.values()]
    stats['avg_tenure_years'] = np.mean(tenures)
    stats['median_tenure_years'] = np.median(tenures)
    stats['min_tenure_years'] = np.min(tenures)
    stats['max_tenure_years'] = np.max(tenures)
    
    # Promotion statistics
    promotion_times = [e['avg_promotion_time_years'] for e in employees.values() if e['avg_promotion_time_years'] is not None]
    if promotion_times:
        stats['avg_promotion_time_years'] = np.mean(promotion_times)
        stats['median_promotion_time_years'] = np.median(promotion_times)
    else:
        stats['avg_promotion_time_years'] = None
        stats['median_promotion_time_years'] = None
    
    # Department breakdown
    dept_counts = defaultdict(int)
    for e in employees.values():
        if pd.notna(e['current_department']):
            dept_counts[e['current_department']] += 1
    stats['department_counts'] = dict(dept_counts)
    
    # Level breakdown
    level_counts = defaultdict(int)
    for e in employees.values():
        if pd.notna(e['current_level']):
            level_counts[e['current_level']] += 1
    stats['level_counts'] = dict(level_counts)
    
    return stats

def main():
    print("="*70)
    print("SALARY HISTORY DATA ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_and_clean_data('data/salary_history.csv')
    print(f"✓ Loaded {len(df)} salary records")
    
    # Calculate employee metrics
    print("\nCalculating employee metrics...")
    employees = calculate_employee_metrics(df)
    print(f"✓ Analyzed {len(employees)} unique employees")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    stats = generate_summary_statistics(employees)
    
    # Print results
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal Employees: {stats['total_employees']}")
    print(f"Active Employees: {stats['active_employees']}")
    print(f"Inactive Employees: {stats['inactive_employees']}")
    print(f"Attrition Rate: {stats['attrition_rate']:.1f}%")
    
    print(f"\nTenure Statistics:")
    print(f"  Average: {stats['avg_tenure_years']:.2f} years")
    print(f"  Median: {stats['median_tenure_years']:.2f} years")
    print(f"  Range: {stats['min_tenure_years']:.2f} - {stats['max_tenure_years']:.2f} years")
    
    if stats['avg_promotion_time_years']:
        print(f"\nPromotion Statistics:")
        print(f"  Average time between promotions: {stats['avg_promotion_time_years']:.2f} years")
        print(f"  Median time between promotions: {stats['median_promotion_time_years']:.2f} years")
    
    print(f"\nDepartment Breakdown:")
    for dept, count in sorted(stats['department_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {dept}: {count}")
    
    print(f"\nLevel Breakdown:")
    for level, count in sorted(stats['level_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level}: {count}")
    
    # Print sample employee profiles
    print("\n" + "="*70)
    print("SAMPLE EMPLOYEE PROFILES")
    print("="*70)
    
    for i, (emp_num, emp) in enumerate(list(employees.items())[:3]):
        print(f"\nEmployee {i+1}: {emp['first_name']} {emp['last_name']}")
        print(f"  Hire Date: {emp['hire_date'].strftime('%Y-%m-%d')}")
        print(f"  Tenure: {emp['tenure_years']:.2f} years")
        print(f"  Status: {emp['status']}")
        print(f"  Attrition: {'Yes' if emp['attrition'] else 'No'}")
        print(f"  Department: {emp['current_department']}")
        print(f"  Level: {emp['current_level']}")
        print(f"  Title: {emp['current_title']}")
        print(f"  Promotions: {emp['num_promotions']}")
        if emp['avg_promotion_time_years']:
            print(f"  Avg Promotion Time: {emp['avg_promotion_time_years']:.2f} years")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return df, employees, stats

if __name__ == "__main__":
    df, employees, stats = main()

