# 🏨 Hotel Booking Cancellations – Machine Learning Project

**Hotel Booking Cancellations** is a machine learning project that predicts whether a hotel booking will be canceled.  
It uses multiple ML models — **Random Forest**, **Decision Tree**, **CatBoost**, **XGBoost**, and **LightGBM** — trained on the *Hotel Booking Cancellations* dataset.  

The pipeline includes **data cleaning**, **feature engineering**, **encoding**, **model training**, and **evaluation**, all built in a modular structure for reusability and clarity.

---

## 📚 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Colab Example](#colab-example)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Contributors](#contributors)

---

## ✨ Features
- Predicts hotel booking cancellations using multiple ML algorithms.
- Automatically downloads dataset if missing.
- Cleans and preprocesses raw data.
- Encodes categorical features and engineers new ones.
- Compares performance of five machine learning models.
- Modular source code design (`src/` folder) for easy expansion.


---

## ⚙️ How It Works

When you run `main.py`, the script:

1. **Checks for the dataset**
   - If `hotel_booking_cancellations.csv` isn’t found, it automatically downloads it from Google Drive using `gdown`.

2. **Preprocesses the data**
   - Cleans missing values  
   - Engineers new features  
   - Encodes categorical variables  

3. **Splits the data**
   - Divides it into training and testing sets.

4. **Trains and evaluates models**
   - Random Forest  
   - Decision Tree  
   - CatBoost  
   - XGBoost  
   - LightGBM  

5. **Outputs performance metrics**
   - Accuracy  
   - ROC-AUC  
   - Classification Report  

---

## 🧠 Colab Example
See the project working interactively in Google Colab:  
👉 [Colab Example](https://colab.research.google.com/drive/1TO9BE5z489NnlbXxF-e_5vzpgx2lQzcg?usp=sharing)

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/hotel-cancellations.git
cd hotel-cancellations

```



2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the project
```bash
python main.py
```
If the dataset isn’t found, it will be downloaded automatically.


## 🧰 Requirements
Python 3.8+

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

catboost

xgboost

lightgbm

gdown



## 👥 Contributors

Developed by Dzhusik & teammate From bootcamp
Feel free to contribute, improve, or fork the project!
