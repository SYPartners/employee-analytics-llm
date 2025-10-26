"""
Analyze the employee data and prepare it for model training.

This script:
1. Loads and analyzes the employee CSV
2. Calculates key metrics (tenure, promotion frequency, attrition)
3. Creates derived features for training
4. Generates summary statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import json

def load_and_clean_data(csv_path):
    """Load the employee CSV and perform initial cleaning."""
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
            # For active employees, calculate to today
            tenure_end_date = datetime.now()
            attrition = False
        else:
            # For inactive employees, use last salary end date or last salary start date
            last_date = emp_data.iloc[-1]['Salary End Date']
            if pd.isna(last_date):
                last_date = emp_data.iloc[-1]['Salary Start Date']
            tenure_end_date = last_date
            attrition = True
        
        tenure_days = (tenure_end_date - hire_date).days
        tenure_years = tenure_days / 365.25
        
        # Count promotions and calculate time in role
        promotions = []
        time_in_roles = []
        
        for i in range(1, len(emp_data)):
            prev_salary = emp_data.iloc[i-1]['Salary']
            curr_salary = emp_data.iloc[i]['Salary']
            prev_date = emp_data.iloc[i-1]['Salary Start Date']
            curr_date = emp_data.iloc[i]['Salary Start Date']
            
            # Calculate time in previous role
            days_in_role = (curr_date - prev_date).days
            time_in_roles.append(days_in_role / 365.25)
            
            # Detect promotion (salary increase > 10%)
            if pd.notna(prev_salary) and pd.notna(curr_salary):
                increase_pct = ((curr_salary - prev_salary) / prev_salary) * 100
                if increase_pct > 10:
                    promotions.append({
                        'date': curr_date,
                        'increase_pct': increase_pct,
                        'from_salary': prev_salary,
                        'to_salary': curr_salary,
                        'time_since_hire_years': (curr_date - hire_date).days / 365.25
                    })
        
        # Calculate time in current role
        current_role_start = emp_data.iloc[-1]['Salary Start Date']
        time_in_current_role = (tenure_end_date - current_role_start).days / 365.25
        
        # Calculate average time between promotions
        if len(promotions) > 1:
            promotion_intervals = []
            for i in range(1, len(promotions)):
                years_between = promotions[i]['time_since_hire_years'] - promotions[i-1]['time_since_hire_years']
                promotion_intervals.append(years_between)
            avg_promotion_time = np.mean(promotion_intervals)
        elif len(promotions) == 1:
            # Time from hire to first promotion
            avg_promotion_time = promotions[0]['time_since_hire_years']
        else:
            avg_promotion_time = None
        
        # Current role information
        current_dept = emp_data.iloc[-1]['Departments']
        current_level = emp_data.iloc[-1]['Level']
        current_step = emp_data.iloc[-1]['Step']
        current_title = emp_data.iloc[-1]['Job Title']
        current_location = emp_data.iloc[-1]['Office Locations']
        current_salary = emp_data.iloc[-1]['Salary']
        
        # Calculate average time in role
        if time_in_roles:
            avg_time_in_role = np.mean(time_in_roles)
        else:
            avg_time_in_role = tenure_years
        
        employees[emp_num] = {
            'employee_number': emp_num,
            'first_name': first_name,
            'last_name': last_name,
            'hire_date': hire_date.strftime('%Y-%m-%d'),
            'tenure_years': round(tenure_years, 2),
            'attrition': attrition,
            'status': current_status,
            'num_promotions': len(promotions),
            'avg_promotion_time_years': round(avg_promotion_time, 2) if avg_promotion_time else None,
            'time_in_current_role_years': round(time_in_current_role, 2),
            'avg_time_in_role_years': round(avg_time_in_role, 2),
            'current_department': current_dept if pd.notna(current_dept) else 'Unknown',
            'current_level': current_level if pd.notna(current_level) else 'Unknown',
            'current_step': current_step if pd.notna(current_step) else 'Unknown',
            'current_title': current_title if pd.notna(current_title) else 'Unknown',
            'current_location': current_location if pd.notna(current_location) else 'Unknown',
            'current_salary': current_salary,
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
        stats['employees_with_promotions'] = len(promotion_times)
    else:
        stats['avg_promotion_time_years'] = None
        stats['median_promotion_time_years'] = None
        stats['employees_with_promotions'] = 0
    
    # Time in role statistics
    time_in_roles = [e['time_in_current_role_years'] for e in employees.values()]
    stats['avg_time_in_current_role_years'] = np.mean(time_in_roles)
    stats['median_time_in_current_role_years'] = np.median(time_in_roles)
    
    # Department breakdown
    dept_counts = defaultdict(int)
    for e in employees.values():
        dept_counts[e['current_department']] += 1
    stats['department_counts'] = dict(dept_counts)
    
    # Level breakdown
    level_counts = defaultdict(int)
    for e in employees.values():
        level_counts[e['current_level']] += 1
    stats['level_counts'] = dict(level_counts)
    
    # Attrition by department
    dept_attrition = defaultdict(lambda: {'total': 0, 'attrited': 0})
    for e in employees.values():
        dept = e['current_department']
        dept_attrition[dept]['total'] += 1
        if e['attrition']:
            dept_attrition[dept]['attrited'] += 1
    
    stats['attrition_by_department'] = {
        dept: {
            'rate': (data['attrited'] / data['total'] * 100) if data['total'] > 0 else 0,
            'count': data['attrited'],
            'total': data['total']
        }
        for dept, data in dept_attrition.items()
    }
    
    return stats

def save_processed_data(employees, output_path='data/processed_employees.json'):
    """Save processed employee data to JSON."""
    # Convert to list for JSON serialization
    employee_list = list(employees.values())
    
    with open(output_path, 'w') as f:
        json.dump(employee_list, f, indent=2, default=str)
    
    print(f"\n✓ Saved processed data to {output_path}")

def main():
    print("="*70)
    print("EMPLOYEE DATA ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_and_clean_data('data/employee_data.csv')
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
    
    print(f"\nTime in Role Statistics:")
    print(f"  Average time in current role: {stats['avg_time_in_current_role_years']:.2f} years")
    print(f"  Median time in current role: {stats['median_time_in_current_role_years']:.2f} years")
    
    if stats['avg_promotion_time_years']:
        print(f"\nPromotion Statistics:")
        print(f"  Employees with promotions: {stats['employees_with_promotions']}/{stats['total_employees']}")
        print(f"  Average time to promotion: {stats['avg_promotion_time_years']:.2f} years")
        print(f"  Median time to promotion: {stats['median_promotion_time_years']:.2f} years")
    
    print(f"\nDepartment Breakdown:")
    for dept, count in sorted(stats['department_counts'].items(), key=lambda x: x[1], reverse=True):
        attrition_info = stats['attrition_by_department'][dept]
        print(f"  {dept}: {count} employees (Attrition: {attrition_info['rate']:.1f}%)")
    
    print(f"\nLevel Breakdown:")
    for level, count in sorted(stats['level_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {level}: {count}")
    
    # Print sample employee profiles
    print("\n" + "="*70)
    print("SAMPLE EMPLOYEE PROFILES")
    print("="*70)
    
    for i, (emp_num, emp) in enumerate(list(employees.items())[:3]):
        print(f"\nEmployee {i+1}: {emp['first_name']} {emp['last_name']}")
        print(f"  Employee Number: {emp['employee_number']}")
        print(f"  Hire Date: {emp['hire_date']}")
        print(f"  Tenure: {emp['tenure_years']} years")
        print(f"  Status: {emp['status']}")
        print(f"  Attrition: {'Yes' if emp['attrition'] else 'No'}")
        print(f"  Department: {emp['current_department']}")
        print(f"  Level: {emp['current_level']}")
        print(f"  Step: {emp['current_step']}")
        print(f"  Title: {emp['current_title']}")
        print(f"  Location: {emp['current_location']}")
        print(f"  Promotions: {emp['num_promotions']}")
        print(f"  Time in current role: {emp['time_in_current_role_years']} years")
        if emp['avg_promotion_time_years']:
            print(f"  Avg time to promotion: {emp['avg_promotion_time_years']} years")
    
    # Save processed data
    save_processed_data(employees)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return df, employees, stats

if __name__ == "__main__":
    df, employees, stats = main()

