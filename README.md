# Telecom Customer Churn Analysis Dashboard

An interactive data analysis dashboard built with **Python + Streamlit** that explores customer churn patterns in a telecom dataset. The dashboard is designed to be clear, explainable, and presentation-ready.

---

## What It Does

This app has **4 dashboard pages**, each focused on a different angle of the churn problem:

| Page | What It Covers |
|------|----------------|
| Overview | Total customers, churn rate, revenue at risk, churn by tenure and internet service |
| Contracts & Payments | How contract type and payment method relate to churn |
| Add-On Features | Which optional services (Security, Streaming, etc.) reduce churn |
| Customer Segments | Machine learning clusters to group customers by behaviour |

---

## Project Structure

```
telecom-churn-dashboard/
│
├── app.py                  ← Main Streamlit app
├── data/
│   └── Telcom_Dataset.csv  ← Dataset (place it here)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-dashboard.git
cd telecom-churn-dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place your `Telcom_Dataset.csv` file inside the `data/` folder:
```
data/Telcom_Dataset.csv
```

> **Dataset source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Requirements

```
streamlit
pandas
plotly
scikit-learn
```

Install all at once:
```bash
pip install streamlit pandas plotly scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## Dashboard Pages — Explained

### Page 1: Overview
**Goal:** Get a quick snapshot of the entire customer base.

- **KPI Cards** — Show total customers, churn rate, average monthly charge, and total monthly revenue at risk from churned customers.
- **Churn Rate by Tenure Group** — A bar chart grouping customers by how many months they've been subscribed. The key finding is that customers who leave in the first 10 months represent the highest churn risk.
- **Churn Rate by Internet Service** — Compares churn across DSL, Fiber Optic, and No Internet. Fiber Optic users churn at ~41% despite paying more, suggesting a service quality problem.

---

### Page 2: Contracts & Payments
**Goal:** Understand how billing structure drives churn decisions.

- **Churn by Contract Type** — Month-to-month customers churn at ~42%; two-year contract customers at only ~3%. The commitment level directly predicts loyalty.
- **Churn by Payment Method** — Electronic check users churn at ~45%, while auto-pay users (bank transfer, credit card) churn at ~15–18%. Passive billing = lower cancellation intent.
- **Monthly Charges Box Plot** — Surprisingly, churned customers pay *more* per month than retained ones, suggesting a value-for-money problem for high-paying customers.

---

### Page 3: Add-On Features
**Goal:** Identify which optional services actually retain customers.

- **Add-On vs No Add-On Churn Comparison** — Grouped bar chart showing churn rates for subscribers vs non-subscribers of each add-on. OnlineSecurity and TechSupport cut churn nearly in half.
- **Adoption Profile: Churned vs Retained** — Retained customers adopt significantly more add-ons across every service. The gap is largest for protective services (security, support).

**Strategic implication:** Bundle OnlineSecurity + TechSupport as a discounted "Peace of Mind Package" for at-risk customers.

---

### Page 4: Customer Segments
**Goal:** Use machine learning to find distinct customer groups and tailor strategy to each.

**What is KMeans Clustering?**
KMeans is an unsupervised machine learning algorithm. It groups customers based on numerical similarities — in this case, **tenure**, **monthly charges**, and **total charges**. We told it to find 4 groups, and it figured out the groupings on its own. We then named the groups based on their characteristics.

| Segment | Description | Strategy |
|---------|-------------|----------|
| Loyal High-Value | Long tenure, high spend, low churn | Retain & Reward |
| Growth Potential | Mid-tenure, moderate spend | Upsell to annual plans |
| New & Uncertain | Short tenure, variable spend | Onboard & Convert |
| At-Risk | High churn rate, likely short tenure | Win-back campaigns |

- **Segment Size Pie Chart** — Shows how the customer base is distributed across segments.
- **Churn Rate Bar Chart** — Compares churn rates across the 4 segments.
- **Scatter Plot (Tenure vs Monthly Charges)** — Each dot is a customer, colored by segment. Shows how the clusters naturally separate.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Streamlit | Web app framework (no HTML/JS needed) |
| Pandas | Data loading and manipulation |
| Plotly | Interactive charts |
| Scikit-learn | KMeans clustering (machine learning) |

---

## Dataset

- **Source:** [Telco Customer Churn — IBM Sample Dataset via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** ~7,043 customers
- **Key columns:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `Churn`, `InternetService`, `PaymentMethod`, and various add-on service columns

---

## Author

**Shamitr Mardikar**  


---

## License

This project is open-source and available under the [MIT License](LICENSE).
