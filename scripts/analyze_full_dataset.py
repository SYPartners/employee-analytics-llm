"""
Comprehensive analysis of the full Namely dataset.

This script:
1. Loads and cleans the complete employee dataset
2. Calculates tenure, promotion patterns, and attrition for each employee
3. Generates training data for the LLM
4. Produces summary statistics and insights
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
    df['Salary Start Date'] = pd.to_datetime(df['Salary Start Date'], format='%m/%d/%Y', errors='coerce')
    df['Salary End Date'] = pd.to_datetime(df['Salary End Date'], format='%m/%d/%Y', errors='coerce')
    df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y', errors='coerce')
    
    # Convert salary to numeric
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    
    # Sort by employee and salary start date
    df = df.sort_values(['Employee Number', 'Salary Start Date'])
    
    return df

def calculate_employee_metrics(df):
    """Calculate key metrics for each employee."""
    employees = []
    
    for emp_num, emp_data in df.groupby('Employee Number'):
        if pd.isna(emp_num):
            continue
            
        emp_data = emp_data.sort_values('Salary Start Date')
        
        # Basic info
        first_name = emp_data.iloc[0]['First name']
        last_name = emp_data.iloc[0]['Last name']
        hire_date = emp_data.iloc[0]['Start date']
        current_status = emp_data.iloc[-1]['User status']
        
        if pd.isna(hire_date):
            continue
        
        # Calculate tenure
        if current_status == 'Active Employee':
            tenure_end_date = datetime.now()
            attrition = False
        else:
            last_date = emp_data.iloc[-1]['Salary End Date']
            if pd.isna(last_date):
                last_date = emp_data.iloc[-1]['Salary Start Date']
            if pd.isna(last_date):
                continue
            tenure_end_date = last_date
            attrition = True
        
        tenure_days = (tenure_end_date - hire_date).days
        tenure_years = tenure_days / 365.25
        
        # Count promotions
        promotions = []
        time_in_roles = []
        
        for i in range(1, len(emp_data)):
            prev_salary = emp_data.iloc[i-1]['Salary']
            curr_salary = emp_data.iloc[i]['Salary']
            prev_date = emp_data.iloc[i-1]['Salary Start Date']
            curr_date = emp_data.iloc[i]['Salary Start Date']
            
            if pd.isna(prev_date) or pd.isna(curr_date):
                continue
            
            # Calculate time in previous role
            days_in_role = (curr_date - prev_date).days
            if days_in_role > 0:
                time_in_roles.append(days_in_role / 365.25)
            
            # Detect promotion (salary increase > 10%)
            if pd.notna(prev_salary) and pd.notna(curr_salary) and prev_salary > 0:
                increase_pct = ((curr_salary - prev_salary) / prev_salary) * 100
                if increase_pct > 10:
                    promotions.append({
                        'date': curr_date.strftime('%Y-%m-%d'),
                        'increase_pct': round(increase_pct, 2),
                        'from_salary': float(prev_salary),
                        'to_salary': float(curr_salary),
                        'time_since_hire_years': round((curr_date - hire_date).days / 365.25, 2)
                    })
        
        # Calculate time in current role
        current_role_start = emp_data.iloc[-1]['Salary Start Date']
        if pd.notna(current_role_start):
            time_in_current_role = (tenure_end_date - current_role_start).days / 365.25
        else:
            time_in_current_role = 0
        
        # Calculate average time between promotions
        if len(promotions) > 1:
            promotion_intervals = []
            for i in range(1, len(promotions)):
                years_between = promotions[i]['time_since_hire_years'] - promotions[i-1]['time_since_hire_years']
                promotion_intervals.append(years_between)
            avg_promotion_time = np.mean(promotion_intervals)
        elif len(promotions) == 1:
            avg_promotion_time = promotions[0]['time_since_hire_years']
        else:
            avg_promotion_time = None
        
        # Current role information
        current_dept = emp_data.iloc[-1]['Departments']
        current_level = emp_data.iloc[-1]['Level']
        current_step = emp_data.iloc[-1]['Step']
        current_title = emp_data.iloc[-1]['Job Title']
        current_location = emp_data.iloc[-1]['Office Locations']
        current_division = emp_data.iloc[-1]['Divisions']
        current_salary = emp_data.iloc[-1]['Salary']
        
        # Calculate average time in role
        if time_in_roles:
            avg_time_in_role = np.mean(time_in_roles)
        else:
            avg_time_in_role = tenure_years
        
        employees.append({
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
            'current_division': current_division if pd.notna(current_division) else 'Unknown',
            'current_salary': float(current_salary) if pd.notna(current_salary) else None,
            'promotion_history': promotions,
            'salary_history_count': len(emp_data)
        })
    
    return employees

def generate_summary_statistics(employees):
    """Generate summary statistics from employee data."""
    stats = {
        'total_employees': len(employees),
        'active_employees': sum(1 for e in employees if e['status'] == 'Active Employee'),
        'inactive_employees': sum(1 for e in employees if e['status'] != 'Active Employee'),
        'attrition_rate': sum(1 for e in employees if e['attrition']) / len(employees) * 100,
    }
    
    # Tenure statistics
    tenures = [e['tenure_years'] for e in employees]
    stats['avg_tenure_years'] = np.mean(tenures)
    stats['median_tenure_years'] = np.median(tenures)
    stats['min_tenure_years'] = np.min(tenures)
    stats['max_tenure_years'] = np.max(tenures)
    
    # Promotion statistics
    promotion_times = [e['avg_promotion_time_years'] for e in employees if e['avg_promotion_time_years'] is not None]
    if promotion_times:
        stats['avg_promotion_time_years'] = np.mean(promotion_times)
        stats['median_promotion_time_years'] = np.median(promotion_times)
        stats['employees_with_promotions'] = len(promotion_times)
    else:
        stats['avg_promotion_time_years'] = None
        stats['median_promotion_time_years'] = None
        stats['employees_with_promotions'] = 0
    
    # Time in role statistics
    time_in_roles = [e['time_in_current_role_years'] for e in employees]
    stats['avg_time_in_current_role_years'] = np.mean(time_in_roles)
    stats['median_time_in_current_role_years'] = np.median(time_in_roles)
    
    # Department breakdown
    dept_counts = defaultdict(int)
    for e in employees:
        dept_counts[e['current_department']] += 1
    stats['department_counts'] = dict(sorted(dept_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Level breakdown
    level_counts = defaultdict(int)
    for e in employees:
        level_counts[e['current_level']] += 1
    stats['level_counts'] = dict(sorted(level_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Location breakdown
    location_counts = defaultdict(int)
    for e in employees:
        location_counts[e['current_location']] += 1
    stats['location_counts'] = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Attrition by department
    dept_attrition = defaultdict(lambda: {'total': 0, 'attrited': 0})
    for e in employees:
        dept = e['current_department']
        dept_attrition[dept]['total'] += 1
        if e['attrition']:
            dept_attrition[dept]['attrited'] += 1
    
    stats['attrition_by_department'] = {
        dept: {
            'rate': round((data['attrited'] / data['total'] * 100), 1) if data['total'] > 0 else 0,
            'count': data['attrited'],
            'total': data['total']
        }
        for dept, data in sorted(dept_attrition.items(), key=lambda x: x[1]['total'], reverse=True)[:15]
    }
    
    return stats

def save_processed_data(employees, output_path='data/processed_employees_full.json'):
    """Save processed employee data to JSON."""
    with open(output_path, 'w') as f:
        json.dump(employees, f, indent=2, default=str)
    
    print(f"\n✓ Saved processed data to {output_path}")

def main():
    print("="*70)
    print("FULL EMPLOYEE DATASET ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_and_clean_data('data/namely_full_dataset.csv')
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
    
    print(f"\nTop 15 Departments:")
    for dept, count in list(stats['department_counts'].items())[:15]:
        attrition_info = stats['attrition_by_department'].get(dept, {'rate': 0})
        print(f"  {dept}: {count} employees (Attrition: {attrition_info['rate']:.1f}%)")
    
    print(f"\nLevel Breakdown:")
    for level, count in stats['level_counts'].items():
        print(f"  {level}: {count}")
    
    print(f"\nTop 10 Locations:")
    for location, count in list(stats['location_counts'].items())[:10]:
        print(f"  {location}: {count}")
    
    # Save processed data
    save_processed_data(employees)
    
    # Save summary stats
    with open('data/summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✓ Saved summary statistics to data/summary_statistics.json")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return df, employees, stats

if __name__ == "__main__":
    df, employees, stats = main()

