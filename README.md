# House Price Prediction (Linear Regression)

## 1. Problem Description
This project aims to predict house prices in Tehran based on property size, number of rooms, available amenities (parking, warehouse, and elevator), and location using a **Linear Regression** model.

**Dataset:** [House Price (Tehran, Iran)](https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran)

---

## 2. Project Structure

```text
house-price-prediction/
│
├── data/
│   ├── raw.csv
│   └── house_prices_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_modeling.ipynb            # Model training and evaluation
│
├── src/
│   ├── data_preprocessing.py
│   └── train.py
│
├── models/
│   └── linear_regression_best.pkl
│
├── images/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_coefficients.png
│
├── requirements.txt
└── README.md
```

## 3. Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Workflow

### 4.1 Exploratory Data Analysis (EDA)

- Detected invalid values in the **Area** column (extremely large values caused by data entry errors) and removed them.
- Both **Price** and **Area** exhibited strong right-skewed distributions, so a **log transformation** was applied.
- The **Address** feature contained more than 190 unique categories with substantial price differences. Rare addresses were grouped into **"Other"** to reduce dimensionality.
- No severe multicollinearity was observed overall, although **Room** and **Area** showed a relatively high correlation. Therefore, two model versions were evaluated: one including **Room** and one excluding it.

### 4.2 Data Preprocessing

- **Target:** `Price_log`
- **Numerical Features:** `Area_log` (+ optional `Room`)
- **Boolean Features:** `Parking`, `Warehouse`, `Elevator`
- **Categorical Feature:** `Address` (One-Hot Encoding with rare categories grouped into **"Other"**)

### 4.3 Model Performance

| Model Version | MAE | RMSE | R² |
|---------------|----:|-----:|---:|
| With Room | 0.2621 | 0.3998 | 0.8637 |
| Without Room | 0.2613 | 0.4049 | 0.8603 |

> **Note:** All evaluation metrics are calculated on the **log-transformed target (`Price_log`)**, not on the original house prices in USD.

---

## 5. Results

- The model achieved an **R² score of approximately 0.86**, indicating that the selected features explain a large proportion of the variance in house prices.
- The two model versions produced very similar results, suggesting that **Room** contributes little additional information beyond **Area**.
- **Area_log** and **Address** were identified as the most influential features for predicting house prices.

---

## 6. Future Improvements

- Experiment with **Ridge Regression** and **Lasso Regression**.
- Compare the performance with non-linear models such as **Random Forest** and **XGBoost**.
- Use **k-fold cross-validation** instead of a single train/test split for a more reliable evaluation.