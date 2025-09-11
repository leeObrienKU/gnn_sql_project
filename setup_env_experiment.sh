#!/bin/bash
set -e

echo "Setting up environment for GNN-SQL experiments..."

# Function to print section headers
print_header() {
    echo
    echo "======================================"
    echo "$1"
    echo "======================================"
}

print_header "Installing system packages..."
apt-get update -qq
apt-get install -y postgresql postgresql-contrib postgresql-client wget

print_header "Installing Python packages..."
pip install --quiet torch torch-geometric psycopg2-binary pandas networkx tabulate wandb matplotlib

# Install PyG optional GPU acceleration libs
pip install --quiet pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)").html

print_header "Setting up PostgreSQL..."
service postgresql start

print_header "Downloading database dump..."
cd /content/gnn_sql_project
wget -O data/empdb.dump https://raw.githubusercontent.com/leeObrienKU/SQL-GNN/main/data/empdb.dump

print_header "Setting up database..."
sudo -u postgres createdb empdb
sudo -u postgres pg_restore -d empdb data/empdb.dump

print_header "Configuring PostgreSQL authentication..."
cat <<EOF > /etc/postgresql/14/main/pg_hba.conf
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

print_header "Restarting PostgreSQL..."
service postgresql restart

print_header "Setting file permissions..."
# Make all scripts executable
chmod 777 *.sh
chmod 777 test_attrition_*.sh

# Create experiment logs directory with full permissions
mkdir -p experiment_logs
chmod 777 experiment_logs

# Ensure data directory is accessible
chmod 777 data
chmod 777 data/*

print_header "Setup complete!"
echo "You can now run the experiments using:"
echo "  • GCN:      !bash test_attrition_gcn.sh"
echo "  • GAT:      !bash test_attrition_gat.sh"
echo "  • GraphSAGE: !bash test_attrition_graphsage.sh"
echo
echo "Results will be saved in the experiment_logs directory."
