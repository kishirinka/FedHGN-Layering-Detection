# FedHGN-Layering-Detection

A Federated Heterogeneous Graph Neural Network (FedHGN) implementation for detecting money laundering patterns in financial transaction networks. This project uses federated learning to enable multiple banks to collaboratively train fraud detection models while preserving data privacy.

## 🎯 Overview

Money laundering detection is a critical challenge in financial institutions. Traditional centralized approaches require sharing sensitive transaction data, which raises privacy and regulatory concerns. This project implements FedHGN, a federated learning framework that allows banks to:

- **Collaboratively train** fraud detection models without sharing raw transaction data
- **Preserve privacy** by keeping sensitive financial data local to each institution
- **Leverage heterogeneous graph structures** to capture complex relationships between accounts, entities, and banks
- **Detect layering patterns** in money laundering schemes

## 🏗️ Architecture

The system models financial transactions as heterogeneous graphs with:
- **Nodes**: Accounts, Banks, Entities
- **Edges**: Transfer relationships, ownership relationships
- **Features**: Transaction amounts, timestamps, payment formats, fraud labels

### Key Components:
- **FedHGN Framework**: Federated learning coordination with alignment regularization
- **RGCN Model**: Relational Graph Convolutional Networks for heterogeneous graphs
- **Data Processing**: Automated graph construction from transaction data
- **Privacy Preservation**: Local training with secure aggregation

## 📁 Project Structure

```
FedHGN-Layering-Detection/
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── configs.yaml                    # Configuration for datasets, models, and FL frameworks
├── FedHGN_Fraud.py                 # Main federated learning implementation
├── fraud.ipynb                     # Interactive fraud detection notebook
├── preprocessing-data.ipynb        # Data preprocessing and graph construction
├── README.md                       # This file
└── data/
    ├── csv/                        # Raw transaction data by bank
    │   ├── df_Arbor_Savings.csv
    │   ├── df_East.csv
    │   ├── df_Japan.csv
    │   ├── df_Laramie.csv
    │   └── df_Oasis_Thrift.csv
    └── graph/                      # Preprocessed graph data
        ├── graph_bank_1.bin
        ├── graph_bank_10.bin
        ├── graph_bank_12.bin
        ├── graph_bank_15.bin
        └── graph_bank_70.bin
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.1.1
- DGL 2.1.0
- NumPy 1.26.4

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/kishirinka/FedHGN-Layering-Detection.git
cd FedHGN-Layering-Detection
```

2. **Install dependencies**:
```bash
# Using requirements.txt (recommended)
pip install -r requirements.txt

# Or install manually with specific versions
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install numpy==1.26.4
pip install torchdata==0.6.1
pip install dgl==2.1.0
pip install scikit-learn scipy pyyaml tqdm pandas
```

### Usage

#### 1. Data Preprocessing
Run the preprocessing notebook to convert raw transaction data into graph format:
```bash
jupyter notebook preprocessing-data.ipynb
```

This will:
- Load transaction data from CSV files
- Create heterogeneous graphs for each bank
- Save preprocessed graphs to `data/graph/`

#### 2. Federated Training
Execute the main federated learning script:
```bash
python FedHGN_Fraud.py
```

Or use the interactive notebook:
```bash
jupyter notebook fraud.ipynb
```

#### 3. Configuration
Modify `configs.yaml` to adjust:
- **Dataset parameters**: task type, data paths
- **Model settings**: hidden dimensions, learning rates, dropout
- **Federated learning**: number of clients, local epochs, aggregation strategy

## 📊 Data Format

### Input Data (CSV)
Each bank's transaction data should contain:
- `Account`: Source account ID
- `Account.1`: Destination account ID  
- `Amount Paid`: Transaction amount
- `Payment Format`: Payment method
- `Timestamp`: Transaction timestamp
- `Entity ID`: Associated entity identifier
- `From Bank`: Originating bank
- `Is Laundering`: Fraud label (0/1)

### Graph Structure
The system creates heterogeneous graphs with:
- **Account nodes**: Individual financial accounts
- **Bank nodes**: Financial institutions
- **Entity nodes**: Account owners/entities
- **Transfer edges**: Account-to-account transactions
- **Belongs_to edges**: Account-to-bank relationships
- **Represents edges**: Account-to-entity relationships

## 🔧 Model Configuration

### Supported Models
- **RGCN**: Relational Graph Convolutional Network
- **RGCN_nc**: RGCN for node classification tasks

### Federated Learning Frameworks
- **FedHGN**: Federated learning with alignment regularization
- **FedAvg**: Standard federated averaging
- **FedProx**: Federated learning with proximal term
- **Local**: Local training only
- **Central**: Centralized training

### Key Hyperparameters
```yaml
models:
  RGCN:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.0
    lr: 0.01
    
frameworks:
  FedHGN:
    num_local_epochs: 3
    max_rounds: 1000
    align_reg: 0.5
    fraction: 1.0
```

## 📈 Results and Evaluation

The model evaluates fraud detection performance using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Fraud detection precision and recall
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

Results are logged during training and saved to the specified output directory.

## 📚 References

- Zhang, C., et al. "FedHGN: A Federated Framework for Heterogeneous Graph Neural Networks." 
- Hamilton, W., et al. "Inductive Representation Learning on Large Graphs."
- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data."

## 📧 Contact

**Author**: kishirinka  
**Repository**: [FedHGN-Layering-Detection](https://github.com/kishirinka/FedHGN-Layering-Detection)

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This implementation is for research and educational purposes. Ensure compliance with financial regulations and privacy laws when working with real transaction data.