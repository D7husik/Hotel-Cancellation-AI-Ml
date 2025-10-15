# Hotel-Cancellation-AI-Ml

🏨 Hotel Booking Cancellations – Machine Learning Project

This project predicts whether a hotel booking will be canceled using multiple machine learning models :
  -Random Forest 
  -Decision Tree
  -CatBoost 
  -XGBoost 
  -LightGBM.

It includes data cleaning, feature engineering, encoding, model training, and evaluation, all organized in a modular structure for clarity and reusability.

Project Structure
hotel_cancellations/
│
├── data/                             # dataset folder (auto-downloaded if missing)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # data loading, cleaning, feature engineering
│   ├── modeling.py                   # ML models and training functions
│
├── main.py                           # main script that runs the entire pipeline
├── requirements.txt                  # dependencies
└── README.md

⚙️ How It Works

When you run main.py, the script will:

1.Check for the dataset
  If not found, it automatically downloads
  hotel_booking_cancellations.csv from Google Drive using gdown.
2.Preprocess the data
  Cleans missing values
  Creates new engineered features
  Encodes categorical variables
3.Split the data into train/test sets.
4.Train and evaluate models:
  Random Forest
  Decision Tree
  CatBoost
  XGBoost
  LightGBM

5.Outputs model performance (accuracy, ROC-AUC, classification report, etc.)


🧠 Colab Example

To see the project logic and visualizations interactively, check out the Colab version:
👉 (https://colab.research.google.com/drive/1TO9BE5z489NnlbXxF-e_5vzpgx2lQzcg?usp=sharing)


🚀 Quick Start
1️⃣ Clone the repo
    git clone https://github.com/yourusername/hotel-cancellations.git
    cd hotel-cancellations

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the project
python main.py


If the dataset isn’t found, it will automatically download from Google Drive.


🧰 Requirements

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

