import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Drop company
    df = df.drop(columns=['company'], errors='ignore')
    # fill children
    df['children'] = df['children'].fillna(0)
    # map hotel
    df['hotel'] = df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
    # map arrival month
    df['arrival_date_month'] = df['arrival_date_month'].map({
        'January':1, 'February':2, 'March':3, 'April':4, 'May':5,
        'June':6, 'July':7, 'August':8, 'September':9,
        'October':10, 'November':11, 'December':12
    })
    # drop null country rows
    df = df.dropna(subset=['country'])
    # feature functions
    df['is_family'] = df.apply(lambda row: 1 if ((row['adults'] > 0) and (row['children'] > 0 or row['babies'] > 0)) else 0, axis=1)
    df['total_customer'] = df['adults'] + df['children'] + df['babies']
    df['deposit_given'] = df['deposit_type'].apply(lambda x: 0 if x in ['No Deposit', 'Refundable'] else 1)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    # drop unused columns
    df = df.drop(columns=[
        'adults', 'children', 'babies', 'deposit_type',
        'reservation_status_date', 'reservation_status'
    ], errors='ignore')
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    # one-hot for these categorical columns
    df = pd.get_dummies(
        df,
        columns=['meal', 'market_segment', 'distribution_channel',
                 'reserved_room_type', 'assigned_room_type', 'customer_type'],
        drop_first=False
    )
    # encode country
    df['country'] = le.fit_transform(df['country'])
    return df
