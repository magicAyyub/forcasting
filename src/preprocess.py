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
        'Disponibilite_de_service': 'first',
        'bug_technique': 'max'
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

    # bug_technique marks a known technical bug depressing volume independently of
    # service outages. Treat missing values as "no bug" (0.0).
    if daily_df['bug_technique'].isnull().sum() > 0:
        print(f"Warning: {daily_df['bug_technique'].isnull().sum()} nulls found in bug_technique. Imputing with 0.0 (No bug).")
        daily_df['bug_technique'] = daily_df['bug_technique'].fillna(0.0)

    # Sort by date
    daily_df = daily_df.sort_values(by=['date', 'canal_verification']).reset_index(drop=True)

    if output_path:
        daily_df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")

    return daily_df

def aggregate_to_monthly(daily_df, output_path=None):
    """Aggregate daily preprocessed data to monthly totals and averages for macro modeling."""
    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    daily_total = daily_df.groupby('date').agg({
        't_verified_canal_sum': 'sum',
        'ferie': 'first',
        'week_end': 'first',
        'Disponibilite_de_service': 'first',
        'bug_technique': 'max'
    }).reset_index()

    daily_total['is_working_day'] = ((daily_total['week_end'] == 0) & (daily_total['ferie'] == 0)).astype(int)

    monthly_df = daily_total.set_index('date').resample('MS').agg({
        't_verified_canal_sum': 'sum',
        'Disponibilite_de_service': 'mean',
        'ferie': 'sum',
        'is_working_day': 'sum',
        'bug_technique': 'max'
    }).reset_index()

    monthly_df['month'] = monthly_df['date'].dt.month
    monthly_df['year'] = monthly_df['date'].dt.year

    if output_path:
        monthly_df.to_csv(output_path, index=False)
        print(f"Monthly preprocessed data saved to {output_path}")

    return monthly_df

if __name__ == '__main__':
    input_file = 'data/raw/Acquisition_LIN_V3.csv'
    output_file = 'data/processed/Acquisition_LIN_clean.csv'
    monthly_file = 'data/processed/Acquisition_LIN_monthly.csv'
    clean_df = load_and_preprocess(input_file, output_file)
    monthly_df = aggregate_to_monthly(clean_df, monthly_file)
    print("Preprocessing completed.")
    print("Daily shape:", clean_df.shape)
    print("Monthly shape:", monthly_df.shape)


