from src.preprocessing import load_data, clean_and_engineer, encode_features
from src.modeling import (split_data, random_forest, decision_tree, 
                          catboost_model, xgboost_model, lightgbm_model)

def main():
    
    try:
        df = load_data('data/hotel_booking_cancellations.csv')
    except FileNotFoundError:
        import os
        os.system("gdown 1sE0if28EBi8REKa8w0mFiA2XH8udIL1C -O data/hotel_booking_cancellations.csv")
        df = load_data('data/hotel_booking_cancellations.csv')
    
    # Preprocess & encode
    df = clean_and_engineer(df)
    df = encode_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train & evaluate
    rf = random_forest(X_train, X_test, y_train, y_test)
    dt = decision_tree(X_train, X_test, y_train, y_test)
    cb = catboost_model(X_train, X_test, y_train, y_test)
    xb = xgboost_model(X_train, X_test, y_train, y_test)
    lgb = lightgbm_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
