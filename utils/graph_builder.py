import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import Dict, Optional

def standardize(df, columns):
    """Standardize numeric columns to zero mean and unit variance"""
    df_std = df.copy()
    for col in columns:
        mean = df_std[col].mean()
        std = df_std[col].std()
        if std > 0:
            df_std[col] = (df_std[col] - mean) / std
        else:
            df_std[col] = 0.0
    return df_std

def get_latest_by(df, by_cols, sort_cols, keep_cols):
    """Get the latest record for each group, avoiding column duplication"""
    # Make sure by_cols are not in keep_cols to avoid duplication
    keep_cols_unique = [col for col in keep_cols if col not in by_cols]
    # Get the latest records
    latest = df.sort_values(sort_cols).groupby(by_cols)[keep_cols_unique].last()
    # Only reset_index if we have by_cols that aren't in the result
    if any(col not in keep_cols_unique for col in by_cols):
        latest = latest.reset_index()
    return latest

def create_graph(
    employees,
    departments,
    dept_emp,
    dept_manager,
    titles,
    salaries,
    task: str = "attrition",
    cutoff_date: str = "2000-01-01",
    use_all_history_edges: bool = True
):
    """Create graph for employee attrition prediction.
    
    Args:
        employees: DataFrame with employee info
        departments: DataFrame with department info
        dept_emp: DataFrame with employee-department relationships
        dept_manager: DataFrame with manager info
        titles: DataFrame with employee titles
        salaries: DataFrame with salary info
        task: Task type ("attrition" or "dept")
        cutoff_date: Date for attrition labeling
        use_all_history_edges: Whether to use historical relationships
    """
    print("\nCreating graph...")
    
    # Calculate employee features
    ref_date = pd.to_datetime(cutoff_date)
    employees = employees.copy()
    employees['birth_date'] = pd.to_datetime(employees['birth_date'])
    employees['hire_date'] = pd.to_datetime(employees['hire_date'])
    
    employees['age_years'] = (ref_date - employees['birth_date']).dt.days / 365.25
    employees['tenure_years'] = (ref_date - employees['hire_date']).dt.days / 365.25
    
    # Get latest salary
    print("Processing salary data...")
    latest_salary = get_latest_by(
        salaries,
        by_cols=['emp_no'],
        sort_cols=['to_date'],
        keep_cols=['emp_no', 'salary']
    ).rename(columns={'salary': 'curr_salary'})
    
    # Get latest department
    print("Processing department data...")
    latest_dept = get_latest_by(
        dept_emp,
        by_cols=['emp_no'],
        sort_cols=['to_date'],
        keep_cols=['emp_no', 'dept_no']
    )
    
    # Get latest title
    print("Processing title data...")
    latest_title = get_latest_by(
        titles,
        by_cols=['emp_no'],
        sort_cols=['to_date'],
        keep_cols=['emp_no', 'title']
    )
    latest_title['title_code'] = latest_title['title'].astype('category').cat.codes
    
    # Calculate salary growth
    print("Calculating salary growth...")
    salary_growth = salaries.groupby('emp_no').agg(
        salary_growth=pd.NamedAgg(
            column='salary',
            aggfunc=lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0.0
        )
    ).reset_index()
    
    # Assemble features
    print("Assembling features...")
    emp_feat = employees[['emp_no', 'age_years', 'tenure_years']].copy()
    emp_feat = emp_feat.merge(latest_salary[['emp_no', 'curr_salary']], on='emp_no', how='left')
    emp_feat = emp_feat.merge(salary_growth, on='emp_no', how='left')
    emp_feat = emp_feat.merge(latest_title[['emp_no', 'title_code']], on='emp_no', how='left')
    emp_feat = emp_feat.merge(latest_dept[['emp_no', 'dept_no']], on='emp_no', how='left')
    
    # Fill missing values
    numeric_cols = ['age_years', 'tenure_years', 'curr_salary', 'salary_growth', 'title_code']
    emp_feat[numeric_cols] = emp_feat[numeric_cols].fillna(0.0)
    
    # Create department one-hot encoding
    print("Creating department encoding...")
    dept_list = sorted(departments['dept_no'].unique())
    dept_to_idx = {dept: idx for idx, dept in enumerate(dept_list)}
    
    dept_onehot = np.zeros((len(emp_feat), len(dept_list)), dtype=np.float32)
    for i, dept in enumerate(emp_feat['dept_no']):
        if pd.notna(dept) and dept in dept_to_idx:
            dept_onehot[i, dept_to_idx[dept]] = 1.0
    
    # Standardize numeric features
    print("Standardizing features...")
    emp_feat_std = standardize(emp_feat[numeric_cols], numeric_cols)
    
    # Final features
    emp_features = np.hstack([
        emp_feat_std.values,
        dept_onehot
    ]).astype(np.float32)
    
    # Department features (match employee feature dimension)
    feature_dim = emp_features.shape[1]
    dept_features = np.zeros((len(dept_list), feature_dim), dtype=np.float32)
    # Set one-hot part of department features
    dept_features[:, -len(dept_list):] = np.eye(len(dept_list))
    
    # Combine features
    x = torch.from_numpy(np.vstack([emp_features, dept_features])).contiguous()
    
    # Create edges
    print("\nCreating edges...")
    edge_list = []
    
    # Employee-Department edges
    for emp_idx, (_, row) in enumerate(latest_dept.iterrows()):
        if pd.notna(row['dept_no']) and row['dept_no'] in dept_to_idx:
            dept_idx = dept_to_idx[row['dept_no']] + len(emp_feat)  # offset for department nodes
            edge_list.append([emp_idx, dept_idx])
            edge_list.append([dept_idx, emp_idx])  # make it bidirectional
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).contiguous()
    
    # Create labels
    print("\nCreating labels...")
    cutoff = pd.to_datetime(cutoff_date)
    latest_emp = get_latest_by(
        dept_emp,
        by_cols=['emp_no'],
        sort_cols=['to_date'],
        keep_cols=['emp_no', 'to_date']
    )
    
    # Employee labels (1 = left before cutoff)
    labels = torch.zeros(len(emp_feat) + len(dept_list), dtype=torch.long).contiguous()
    for idx, (_, row) in enumerate(latest_emp.iterrows()):
        if str(row['to_date']) != '9999-01-01':
            if pd.to_datetime(row['to_date']) < cutoff:
                labels[idx] = 1
    
    # Create train/val/test split
    print("\nCreating data splits...")
    num_nodes = len(emp_feat)  # only split employees
    perm = torch.randperm(num_nodes)
    
    train_idx = perm[:int(0.6 * num_nodes)]
    val_idx = perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    test_idx = perm[int(0.8 * num_nodes):]
    
    train_mask = torch.zeros(len(emp_feat) + len(dept_list), dtype=torch.bool).contiguous()
    val_mask = torch.zeros(len(emp_feat) + len(dept_list), dtype=torch.bool).contiguous()
    test_mask = torch.zeros(len(emp_feat) + len(dept_list), dtype=torch.bool).contiguous()
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Create graph
    data = Data(x=x, edge_index=edge_index, y=labels)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Add metadata
    data.num_employees = len(emp_feat)
    data.num_departments = len(dept_list)
    data.num_classes = 2  # binary classification
    data.task = task
    data.ref_date = str(ref_date.date())
    
    print("\nGraph creation complete!")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of employees: {data.num_employees}")
    print(f"Number of departments: {data.num_departments}")
    print(f"Feature dimension: {feature_dim}")
    
    return data