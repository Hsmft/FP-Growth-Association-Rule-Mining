# FP-Growth Association Rule Mining from Scratch

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project features a from-scratch implementation of the **FP-Growth algorithm** in Python to perform association rule mining. The analysis is conducted on the "Adult Census Income" dataset, which requires extensive preprocessing to transform mixed-type attributes into a transactional format suitable for frequent itemset mining.

---

## 📋 Project Workflow

This project is broken down into three main stages, executed by separate scripts:

### 1. Data Exploration (`src/explore.py`)
* Initial analysis of the dataset to understand its structure and characteristics. 
* Identified numerical and categorical attributes. 
* Counted unique values for each attribute.
* Located and quantified missing values across the dataset. 

### 2. Data Preprocessing (`src/preprocess.py`)
The raw dataset contains a mix of attribute types and missing values, which are handled as follows:
* **Missing Value Imputation:** Missing values in categorical features are filled using the mode of each feature. 
* **Merging Categories:** For the `native-country` attribute, countries with fewer than 40 individuals are merged into a single "Others" category to reduce dimensionality. 
* **Categorical Binarization:** All categorical and symmetric binary attributes (like `workclass` and `income`) are converted into asymmetric binary "items" (e.g., `workclass = Federal-gov`, `income >50K`). 
* **Continuous Discretization:** Continuous attributes (`age`, `capital-gain`, etc.) are discretized into intervals using equal-frequency, equal-width, or custom-defined split points and then binarized. 

### 3. Association Analysis (`src/association_analysis.py`)
* **Frequent Itemset Mining:** A from-scratch implementation of the **FP-Growth algorithm** (`find_frequent_itemsets`) is used to efficiently discover frequent itemsets that meet a minimum support count, without candidate generation. 
* **Rule Generation:** A function (`generate_rules`) generates association rules from the frequent itemsets that meet a minimum confidence threshold. 

---

## 🛠️ Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib

---

## 🚀 Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/FP-Growth-Association-Rule-Mining.git](https://github.com/your-username/FP-Growth-Association-Rule-Mining.git)
    cd FP-Growth-Association-Rule-Mining
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the scripts:**
    The analysis is performed in sequence. First, run the preprocessing script to generate the transactional data:
    ```bash
    python src/preprocess.py
    ```
    This will create `dist/adult_preprocessed.csv`. Next, run the association analysis script with a minimum support of 13,000 and minimum confidence of 95%:
    ```bash
    python src/association_analysis.py
    ```
    This script will generate `dist/freq_itemsets.txt` and `dist/rules.txt`. 

---

## 🙏 Acknowledgments
* This analysis uses the **Adult Census Income dataset**, originally from the UCI Machine Learning Repository. 

---

## 📄 License
This project is licensed under the MIT License.
