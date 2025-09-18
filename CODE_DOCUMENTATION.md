# GNN SQL Project - Code Documentation

## 📋 Overview
This document provides comprehensive documentation of all code sections in the GNN SQL project for employee attrition prediction.

---

## 🚀 Main Entry Point

### `main.py` - Main Training Script
**Purpose**: Entry point for training GNN models on employee data

**Key Features**:
- Command-line argument parsing
- Model selection (GCN, GAT, GraphSAGE)
- Weights & Biases integration
- Complete training pipeline

**Key Sections**:
```python
# Argument parsing for model configuration
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'GraphSAGE'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)

# Weights & Biases integration
if args.wandb and _WANDB_AVAILABLE:
    logger.wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={},
        name=f"{args.model}"
    )

# Main training pipeline
test_acc = train_and_evaluate(
    model=model, data=data, train_loader=train_loader,
    epochs=args.epochs, lr=args.lr, logger=logger
)
```

---

## 🧠 Model Architecture

### `models/gnn_model.py` - GNN Model Implementation
**Purpose**: Implements three GNN architectures for employee attrition prediction

**Supported Models**:
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network  
- **GraphSAGE**: Graph Sample and Aggregate

**Key Implementation**:
```python
class GNN(torch.nn.Module):
    def __init__(self, model_type='GCN', input_dim=4, hidden_dim=64, 
                 output_dim=2, num_layers=2, dropout=0.5, num_heads=4):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_heads = num_heads
        
        # Create list to hold all layers
        self.convs = torch.nn.ModuleList()
        
        # Input layer - different for each model type
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, 
                                    heads=self.num_heads, dropout=dropout))
            hidden_dim = hidden_dim * self.num_heads
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Process through all layers except last
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        if self.model_type == 'GAT':
            x = F.elu(x)
            
        return F.log_softmax(x, dim=1)
```

---

## 📊 Data Loading

### `utils/data_loader.py` - Database Connection
**Purpose**: Loads employee data from PostgreSQL database

**Key Features**:
- SQLAlchemy engine for database connection
- Schema-aligned column mapping
- Error handling and fallback options

**Key Implementation**:
```python
def load_employees_db():
    """Load data from PostgreSQL with schema-aligned column names"""
    try:
        # Create SQLAlchemy engine
        engine = create_engine('postgresql+psycopg2://postgres@/empdb?host=/var/run/postgresql')

        # Define queries matching your actual schema
        queries = {
            'employees': """
                SELECT 
                    id AS emp_no,
                    birth_date,
                    first_name,
                    last_name,
                    gender,
                    hire_date
                FROM employees.employee
            """,
            'departments': """
                SELECT 
                    id AS dept_no,
                    dept_name
                FROM employees.department
            """,
            # ... more queries
        }
        
        # Load all tables
        data = {}
        for table_name, query in queries.items():
            data[table_name] = pd.read_sql(query, engine)
        
        return data['employees'], data['departments'], data['dept_emp'], \
               data['dept_manager'], data['titles'], data['salaries']
               
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return None, None, None, None, None, None
```

---

## 🔗 Graph Construction

### `utils/graph_builder.py` - Graph Creation
**Purpose**: Converts relational data into PyTorch Geometric graph format

**Key Features**:
- Employee-department relationship modeling
- Feature engineering for employee nodes
- Temporal data handling
- Train/validation/test splits

**Key Implementation**:
```python
def create_graph(employees, departments, dept_emp, dept_manager, 
                titles, salaries, task="attrition", cutoff_date="2000-01-01",
                use_all_history_edges=True):
    """Create graph for employee attrition prediction"""
    
    # Convert dates
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    # Create employee features
    employee_features = []
    employee_labels = []
    
    for _, emp in employees.iterrows():
        # Basic features
        age_at_hire = (emp['hire_date'] - emp['birth_date']).days / 365.25
        tenure_at_cutoff = (cutoff_dt - emp['hire_date']).days / 365.25
        
        # Department features
        emp_depts = dept_emp[dept_emp['emp_no'] == emp['id']]
        num_dept_changes = len(emp_depts) - 1
        current_dept = emp_depts['dept_no'].iloc[-1] if len(emp_depts) > 0 else 0
        
        # Title features
        emp_titles = titles[titles['emp_no'] == emp['id']]
        num_title_changes = len(emp_titles) - 1
        
        # Salary features
        emp_salaries = salaries[salaries['emp_no'] == emp['id']]
        if len(emp_salaries) > 0:
            current_salary = emp_salaries['amount'].iloc[-1]
            salary_growth = (emp_salaries['amount'].iloc[-1] - 
                           emp_salaries['amount'].iloc[0]) / len(emp_salaries)
        else:
            current_salary = 0
            salary_growth = 0
        
        # Create feature vector
        features = [
            age_at_hire,
            tenure_at_cutoff,
            num_dept_changes,
            num_title_changes,
            current_salary / 1000,  # Normalize salary
            salary_growth / 1000,
            1 if emp['gender'] == 'M' else 0
        ]
        
        employee_features.append(features)
        
        # Create labels for attrition prediction
        if task == "attrition":
            left_before_cutoff = any(
                (dept_emp['emp_no'] == emp['id']) & 
                (dept_emp['to_date'] <= cutoff_dt) & 
                (dept_emp['to_date'] != pd.to_datetime('9999-01-01'))
            )
            employee_labels.append(1 if left_before_cutoff else 0)
        else:
            employee_labels.append(current_dept)
    
    # Convert to tensors
    x = torch.tensor(employee_features, dtype=torch.float)
    y = torch.tensor(employee_labels, dtype=torch.long)
    
    # Create edges (employee-department connections)
    edge_list = []
    for _, row in dept_emp.iterrows():
        if row['emp_no'] in emp_id_to_idx:
            emp_idx = emp_id_to_idx[row['emp_no']]
            dept_idx = len(employees) + row['dept_no'] - 1
            edge_list.append([emp_idx, dept_idx])
            edge_list.append([dept_idx, emp_idx])  # Undirected
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, y=y, edge_index=edge_index, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    return data
```

---

## 🏋️ Training Pipeline

### `models/trainer.py` - Training and Evaluation
**Purpose**: Handles model training, validation, and evaluation

**Key Features**:
- Early stopping
- Learning rate scheduling
- Class weight balancing
- Comprehensive metrics tracking

**Key Implementation**:
```python
def train_and_evaluate(model, data, train_loader, epochs, lr, logger, 
                      pos_threshold=0.5, auto_threshold=False,
                      threshold_mode='fixed', target_precision=None, 
                      target_recall=None, lr_decay_type="exponential",
                      lr_decay_gamma=0.95, lr_decay_step_size=20,
                      patience=20, use_class_weights=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    
    # Learning rate scheduler
    if lr_decay_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)
    elif lr_decay_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=lr_decay_step_size, 
                                                   gamma=lr_decay_gamma)
    
    # Class weights for imbalanced data
    if use_class_weights:
        class_counts = torch.bincount(data.y[data.train_mask])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = torch.nn.NLLLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.NLLLoss()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask]).item()
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), val_pred.cpu())
            val_f1 = f1_score(data.y[data.val_mask].cpu(), val_pred.cpu(), average='weighted')
        
        # Log metrics
        if logger:
            logger.log_metrics(epoch, total_loss / len(train_loader), 
                             val_loss, val_acc, val_f1)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(logger.log_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return test_acc
```

---

## 📊 Data Analysis

### `analyze_data_distribution.py` - Exploratory Data Analysis
**Purpose**: Comprehensive data analysis and visualization

**Key Features**:
- Database connection and querying
- Statistical analysis of employee data
- Visualization generation
- HTML report creation

**Key Sections**:
```python
def connect_to_db():
    """Establish database connection with multiple fallback options"""
    connection_params = [
        {'dbname': 'empdb', 'user': 'postgres', 'host': '/var/run/postgresql'},
        {'dbname': 'empdb', 'user': 'postgres', 'host': 'localhost'},
        {'dbname': 'empdb', 'user': 'postgres', 'host': '127.0.0.1'}
    ]
    
    for params in connection_params:
        try:
            conn = psycopg2.connect(**params)
            print(f"Connected to PostgreSQL at {params['host']}")
            return conn
        except Exception as e:
            continue
    return None

def analyze_attrition_patterns(conn):
    """Analyze employee attrition patterns over time"""
    query = """
        SELECT 
            EXTRACT(YEAR FROM de.from_date) as year,
            COUNT(*) as total_employees,
            COUNT(CASE WHEN de.to_date != '9999-01-01' THEN 1 END) as leavers,
            ROUND(COUNT(CASE WHEN de.to_date != '9999-01-01' THEN 1 END) * 100.0 / COUNT(*), 2) as attrition_rate
        FROM employees.dept_emp de
        GROUP BY EXTRACT(YEAR FROM de.from_date)
        ORDER BY year
    """
    return pd.read_sql(query, conn)

def generate_visualizations(conn, output_dir):
    """Generate comprehensive data visualizations"""
    # Attrition patterns over time
    attrition_data = analyze_attrition_patterns(conn)
    
    # Department metrics
    dept_metrics = analyze_department_metrics(conn)
    
    # Salary analysis
    salary_data = analyze_salary_patterns(conn)
    
    # Generate plots
    plot_attrition_trends(attrition_data, output_dir)
    plot_department_metrics(dept_metrics, output_dir)
    plot_salary_distribution(salary_data, output_dir)
```

---

## 🎨 Visualization

### `utils/plots.py` - Plotting Utilities
**Purpose**: Generate training curves and evaluation plots

**Key Functions**:
```python
def plot_training_curves(metrics, output_dir):
    """Plot training and validation curves"""
    epochs = [m['epoch'] for m in metrics]
    train_losses = [m['train_loss'] for m in metrics]
    val_losses = [m['val_loss'] for m in metrics]
    val_accs = [m['val_accuracy'] for m in metrics]
    val_f1s = [m['val_f1'] for m in metrics]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # ... more plots
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix with normalization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw confusion matrix
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix (Raw)')
    # ... add annotations
    
    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix (Normalized)')
    # ... add annotations
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 📝 Experiment Logging

### `utils/experiment_logger.py` - Results Tracking
**Purpose**: Comprehensive experiment logging and Weights & Biases integration

**Key Features**:
- Automatic directory creation
- Parameter and metrics logging
- Weights & Biases integration
- Model checkpointing

**Key Implementation**:
```python
class ExperimentLogger:
    def __init__(self, project_name="gnn-sql-attrition", entity=None):
        self.project_name = project_name
        self.entity = entity
        self.wandb_run = None
        self.metrics = {"training": [], "validation": []}
        self.log_dir = f"experiment_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_params(self, params):
        """Log parameters to JSON file"""
        params_file = os.path.join(self.log_dir, "parameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_metrics(self, epoch, train_loss, val_loss, val_acc, val_f1):
        """Log training metrics"""
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        }
        
        self.metrics["training"].append(metrics)
        
        if self.wandb_run:
            wandb.log(metrics)
    
    def log_final_metrics(self, test_acc, test_f1, test_precision, 
                         test_recall, confusion_matrix, roc_auc, pr_auc):
        """Log final test metrics"""
        final_metrics = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc
        }
        
        metrics_file = os.path.join(self.log_dir, "experiment_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
```

---

## 🚀 Execution Scripts

### Test Scripts
**Purpose**: Easy execution of different model experiments

**Available Scripts**:
- `test_attrition_gcn.sh` - GCN experiments
- `test_attrition_gat.sh` - GAT experiments  
- `test_attrition_graphsage.sh` - GraphSAGE experiments

**Example Script Content**:
```bash
#!/bin/bash
# test_attrition_gcn.sh

echo "Running GCN experiment..."

python main.py \
    --model GCN \
    --epochs 50 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.5 \
    --lr 0.001 \
    --batch_size 1024 \
    --patience 20 \
    --task attrition \
    --cutoff "2002-12-31" \
    --class_weights \
    --wandb \
    --wandb_project "gnn-sql-attrition"

echo "GCN experiment completed!"
```

---

## 📦 Dependencies

### `requirements.txt`
```
torch>=1.9.0
torch-geometric>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0
scikit-learn>=1.0.0
tabulate>=0.8.9
wandb>=0.12.0
```

---

## 🎯 Usage Examples

### Basic Training
```bash
python main.py --model GCN --epochs 50
```

### Advanced Configuration
```bash
python main.py \
    --model GAT \
    --epochs 100 \
    --hidden_dim 128 \
    --num_layers 3 \
    --dropout 0.3 \
    --num_heads 8 \
    --lr 0.0005 \
    --class_weights \
    --wandb
```

### Data Analysis
```bash
python analyze_data_distribution.py
```

---

## 📊 Results Structure

Each experiment creates a timestamped directory in `experiment_logs/`:
```
experiment_logs/
├── GCN_20250912_090140/
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   ├── experiment_metrics.json
│   ├── parameters.json
│   └── metrics/
│       ├── performance_20250912_090140.txt
│       └── training_metrics.json
├── GAT_20250912_090706/
└── GraphSAGE_20250912_090335/
```

This comprehensive documentation covers all major code sections in the GNN SQL project, providing a complete reference for understanding and extending the codebase.
