import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import Dict, Optional

def standardize(df, columns):
    """Standardize numeric columns to zero mean and unit variance.
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize
    """
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
    keep_cols_unique = [col for col in keep_cols if col not in by_cols]
    latest = df.sort_values(sort_cols).groupby(by_cols)[keep_cols_unique].last()
    if any(col not in keep_cols_unique for col in by_cols):
        latest = latest.reset_index()
    return latest

class GraphBuilder:
    def __init__(self):
        self.emp_features = None
        self.dept_features = None
        self.title_features = None

    def prepare_features(self, employees, departments, dept_emp, titles, salaries, ref_date):
        """Prepare node features"""
        print("Preparing features...")
        ref_date = pd.to_datetime(ref_date)
        
        employees = employees.copy()
        employees['birth_date'] = pd.to_datetime(employees['birth_date'])
        employees['hire_date'] = pd.to_datetime(employees['hire_date'])
        
        employees['age_years'] = (ref_date - employees['birth_date']).dt.days / 365.25
        employees['tenure_years'] = (ref_date - employees['hire_date']).dt.days / 365.25
        
        print("Processing salary data...")
        latest_salary = get_latest_by(
            salaries,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'salary']
        ).rename(columns={'salary': 'curr_salary'})
        
        print("Processing department data...")
        latest_dept = get_latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'dept_no']
        )
        
        print("Processing title data...")
        latest_title = get_latest_by(
            titles,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'title']
        )
        latest_title['title_code'] = latest_title['title'].astype('category').cat.codes
        
        print("Calculating salary growth...")
        salary_growth = salaries.groupby('emp_no').agg(
            salary_growth=pd.NamedAgg(
                column='salary',
                aggfunc=lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0.0
            )
        ).reset_index()
        
        print("Assembling features...")
        emp_feat = employees[['emp_no', 'age_years', 'tenure_years']].copy()
        emp_feat = emp_feat.merge(latest_salary[['emp_no', 'curr_salary']], on='emp_no', how='left')
        emp_feat = emp_feat.merge(salary_growth, on='emp_no', how='left')
        emp_feat = emp_feat.merge(latest_title[['emp_no', 'title_code']], on='emp_no', how='left')
        emp_feat = emp_feat.merge(latest_dept[['emp_no', 'dept_no']], on='emp_no', how='left')
        
        numeric_cols = ['age_years', 'tenure_years', 'curr_salary', 'salary_growth', 'title_code']
        emp_feat[numeric_cols] = emp_feat[numeric_cols].fillna(0.0)
        
        print("Creating department encoding...")
        dept_list = sorted(departments['dept_no'].unique())
        dept_to_idx = {dept: idx for idx, dept in enumerate(dept_list)}
        
        dept_onehot = np.zeros((len(emp_feat), len(dept_list)), dtype=np.float32)
        for i, dept in enumerate(emp_feat['dept_no']):
            if pd.notna(dept) and dept in dept_to_idx:
                dept_onehot[i, dept_to_idx[dept]] = 1.0
        
        print("Standardizing features...")
        emp_feat_std = standardize(emp_feat[numeric_cols], numeric_cols)
        
        self.emp_features = np.hstack([
            emp_feat_std.values,
            dept_onehot
        ]).astype(np.float32)
        
        self.dept_features = np.eye(len(dept_list), dtype=np.float32)
        
        num_titles = len(latest_title['title_code'].unique())
        self.title_features = np.eye(num_titles, dtype=np.float32)
        
        print("Feature preparation complete!")
        return emp_feat['emp_no'].values

    def create_heterogeneous_graph(self, employees, departments, dept_emp, titles, salaries, cutoff_date: str) -> HeteroData:
        """Create heterogeneous graph with employee, department, and title nodes"""
        print("\nCreating heterogeneous graph...")
        
        emp_ids = self.prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        data = HeteroData()
        
        data['employee'].x = torch.from_numpy(self.emp_features)
        data['department'].x = torch.from_numpy(self.dept_features)
        data['title'].x = torch.from_numpy(self.title_features)
        
        print("\nCreating edges...")
        
        latest_dept = get_latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'dept_no']
        )
        
        emp_dept_edges = []
        for emp_idx, (_, row) in enumerate(latest_dept.iterrows()):
            dept_idx = int(row['dept_no'].replace('d', '')) - 1
            emp_dept_edges.append([emp_idx, dept_idx])
        
        if emp_dept_edges:
            emp_dept_edges = torch.tensor(emp_dept_edges, dtype=torch.long).t()
            data['employee', 'works_in', 'department'].edge_index = emp_dept_edges
        
        latest_title = get_latest_by(
            titles,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'title']
        )
        
        title_to_idx = {title: idx for idx, title in enumerate(sorted(latest_title['title'].unique()))}
        emp_title_edges = []
        
        for emp_idx, (_, row) in enumerate(latest_title.iterrows()):
            title_idx = title_to_idx[row['title']]
            emp_title_edges.append([emp_idx, title_idx])
        
        if emp_title_edges:
            emp_title_edges = torch.tensor(emp_title_edges, dtype=torch.long).t()
            data['employee', 'has_role', 'title'].edge_index = emp_title_edges
        
        print("\nCreating labels...")
        cutoff = pd.to_datetime(cutoff_date)
        latest_emp = get_latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'to_date']
        )
        
        labels = torch.zeros(len(emp_ids), dtype=torch.long)
        for idx, (_, row) in enumerate(latest_emp.iterrows()):
            if str(row['to_date']) != '9999-01-01':
                if pd.to_datetime(row['to_date']) < cutoff:
                    labels[idx] = 1
        
        data['employee'].y = labels
        
        print("\nCreating data splits...")
        num_nodes = len(emp_ids)
        perm = torch.randperm(num_nodes)
        
        train_idx = perm[:int(0.6 * num_nodes)]
        val_idx = perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]
        test_idx = perm[int(0.8 * num_nodes):]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        data['employee'].train_mask = train_mask
        data['employee'].val_mask = val_mask
        data['employee'].test_mask = test_mask
        
        print("\nGraph creation complete!")
        print(f"Number of employee nodes: {data['employee'].num_nodes}")
        print(f"Number of department nodes: {data['department'].num_nodes}")
        print(f"Number of title nodes: {data['title'].num_nodes}")
        
        return data

    def create_homogeneous_graph(self, employees, departments, dept_emp, titles, salaries, cutoff_date: str) -> Data:
        """Create homogeneous graph (employee nodes only)"""
        print("\nCreating homogeneous graph...")
        
        emp_ids = self.prepare_features(employees, departments, dept_emp, titles, salaries, cutoff_date)
        
        data = Data(
            x=torch.from_numpy(self.emp_features),
            edge_index=None,
            y=None
        )
        
        print("\nCreating edges...")
        latest_dept = get_latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'dept_no']
        )
        
        edge_list = []
        dept_groups = latest_dept.groupby('dept_no')['emp_no'].apply(list)
        
        for dept_employees in dept_groups:
            if len(dept_employees) > 1:
                dept_indices = []
                for emp_no in dept_employees:
                    try:
                        idx = np.where(emp_ids == emp_no)[0][0]
                        dept_indices.append(idx)
                    except IndexError:
                        continue
                
                for i in range(len(dept_indices)):
                    for j in range(i + 1, len(dept_indices)):
                        edge_list.append([dept_indices[i], dept_indices[j]])
                        edge_list.append([dept_indices[j], dept_indices[i]])
        
        if edge_list:
            data.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        print("\nCreating labels...")
        cutoff = pd.to_datetime(cutoff_date)
        latest_emp = get_latest_by(
            dept_emp,
            by_cols=['emp_no'],
            sort_cols=['to_date'],
            keep_cols=['emp_no', 'to_date']
        )
        
        labels = torch.zeros(len(emp_ids), dtype=torch.long)
        for idx, (_, row) in enumerate(latest_emp.iterrows()):
            if str(row['to_date']) != '9999-01-01':
                if pd.to_datetime(row['to_date']) < cutoff:
                    labels[idx] = 1
        
        data.y = labels
        
        print("\nCreating data splits...")
        num_nodes = len(emp_ids)
        perm = torch.randperm(num_nodes)
        
        train_idx = perm[:int(0.6 * num_nodes)]
        val_idx = perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]
        test_idx = perm[int(0.8 * num_nodes):]
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        print("\nGraph creation complete!")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")

    return data