import pandas as pd
import numpy as np

def load_and_preprocess(filepath, output_path=None):
    # Load data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # 1. Drop 2022 data
    df = df[df['date'].dt.year > 2022]

    # 2. Filter out 'facteur' and 'autre' channels
    df = df[df['canal_verification'].isin(['digital', 'bureau'])]

    # 3. Aggregate daily duplicate records by summing the volume
    # When Disponibilite_de_service, ferie, and week_end are identical within a date for a given channel
    agg_funcs = {
        'ferie': 'first',
        'week_end': 'first',
        't_verified_canal_sum': 'sum',
        'Disponibilite_de_service': 'first'
    }
    daily_df = df.groupby(['date', 'canal_verification']).agg(agg_funcs).reset_index()

    # 4. Engineer calendar features
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['year'] = daily_df['date'].dt.year

    # Map month as categorical variables for modeling later, or keep as integers
    # Let's keep as integers and create dummy variables in the modeling step if needed.

    # 5. Handle Disponibilite_de_service nulls (since we dropped 2022, there should be none, but let's be safe)
    # 0.0 -> Available (Normal), 1.0 -> Outage/Disruption (Drop in volume)
    if daily_df['Disponibilite_de_service'].isnull().sum() > 0:
        print(f"Warning: {daily_df['Disponibilite_de_service'].isnull().sum()} nulls found in availability. Imputing with 0.0 (Normal).")
        daily_df['Disponibilite_de_service'] = daily_df['Disponibilite_de_service'].fillna(0.0)

    # Sort by date
    daily_df = daily_df.sort_values(by=['date', 'canal_verification']).reset_index(drop=True)

    if output_path:
        daily_df.to_csv('data/processed/Acquisition_LIN_clean.csv', index=False)
        print(f"Preprocessed data saved to data/processed/Acquisition_LIN_clean.csv")

    return daily_df

if __name__ == '__main__':
    input_file = 'data/raw/Acquisition_LIN.csv'
    output_file = 'data/processed/Acquisition_LIN_clean.csv'
    clean_df = load_and_preprocess(input_file, output_file)
    print("Preprocessing completed.")
    print(clean_df.head())
    print("\nShape:", clean_df.shape)
