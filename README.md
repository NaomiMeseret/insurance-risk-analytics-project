# 🏥 End-to-End Insurance Risk Analytics & Predictive Modeling

## 📋 Project Overview

This project analyzes historical insurance claim data for **AlphaCare Insurance Solutions (ACIS)** to optimize marketing strategy and discover "low-risk" targets for premium reduction, thereby creating opportunities to attract new clients.

### 🎯 Business Objective

Develop cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa. The analysis focuses on:

- 🎯 Identifying low-risk segments for premium optimization
- 🤖 Building predictive models for optimal premium values
- 🧪 Performing A/B hypothesis testing to validate risk differences
- 📊 Statistical modeling and machine learning for claims prediction

## 📁 Project Structure

```
.
├── data/                  # 📊 Raw and processed data files (tracked with DVC)
├── notebooks/             # 📓 Jupyter notebooks for exploratory analysis
├── src/                   # 💻 Source code modules
│   ├── data/             # 📥 Data loading and preprocessing
│   ├── eda/              # 🔍 Exploratory data analysis scripts
│   ├── models/           # 🤖 Machine learning models
│   └── utils/            # 🛠️ Utility functions
├── tests/                # ✅ Unit tests
├── reports/               # 📈 Generated reports and visualizations
├── .github/              # ⚙️ GitHub Actions workflows
│   └── workflows/
├── .dvc/                 # 🔄 DVC configuration (auto-generated)
├── requirements.txt      # 📦 Python dependencies
├── .gitignore           # 🚫 Git ignore rules
└── README.md            # 📖 This file
```

## 📊 Data Description

**⏰ Time Period:** February 2014 to August 2015 (18 months)

**🔑 Key Data Columns:**

- **📄 Policy Information:** UnderwrittenCoverID, PolicyID, TransactionMonth
- **👤 Client Information:** IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender
- **📍 Location:** Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
- **🚗 Vehicle:** ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet
- **📋 Plan:** SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType
- **💰 Financial:** TotalPremium, TotalClaims

## 🚀 Setup Instructions

### 📋 Prerequisites

- 🐍 Python 3.8+
- 🔀 Git
- 🔄 DVC (Data Version Control)

### 💻 Installation

1. **📥 Clone the repository:**

```bash
git clone <repository-url>
cd "End-to-End Insurance Risk Analytics & Predictive Modeling"
```

2. **🌐 Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **📦 Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **🔄 Initialize DVC:**

```bash
dvc init
```

5. **💾 Set up DVC remote storage:**

```bash
mkdir -p ~/dvc_storage
dvc remote add -d localstorage ~/dvc_storage
```

## 🎮 Usage

### 🔍 Running EDA

```bash
python src/eda/exploratory_analysis.py
```

### 🧪 Running Hypothesis Tests

```bash
python src/hypothesis_testing/ab_tests.py
```

### 🤖 Training Models

```bash
python src/models/train_models.py
```

## 📈 Key Metrics & KPIs

- **📉 Loss Ratio:** TotalClaims / TotalPremium
- **🎯 Risk Segmentation:** By Province, VehicleType, Gender
- **📊 Model Performance:** R², RMSE, MAE for regression models
- **✅ Statistical Significance:** p-values for hypothesis tests

## ❓ EDA Guiding Questions

1. 📊 What is the overall Loss Ratio for the portfolio? How does it vary by Province, VehicleType, and Gender?
2. 📈 What are the distributions of key financial variables? Are there outliers in TotalClaims or CustomValueEstimate?
3. ⏰ Are there temporal trends? Did claim frequency or severity change over the 18-month period?
4. 🚗 Which vehicle makes/models are associated with the highest and lowest claim amounts?

## 🧪 Hypothesis Tests

- **H₀:** 🌍 There are no risk differences across provinces
- **H₀:** 📮 There are no risk differences between zipcodes
- **H₀:** 💰 There is no significant margin (profit) difference between zip codes
- **H₀:** 👥 There is no significant risk difference between Women and men

## 🎓 Learning Outcomes

- 🔧 Data Engineering (DE)
- 🔮 Predictive Analytics (PA)
- 🤖 Machine Learning Engineering (MLE)
- 📊 Statistical Modeling and Analysis
- 🧪 A/B Testing Design and Implementation
- 🔄 Data Versioning with DVC
