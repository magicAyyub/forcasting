import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import evaluate_model

# 2026 French Holidays
FERIE_2026 = [
    '2026-01-01', '2026-04-06', '2026-05-01', '2026-05-08',
    '2026-05-14', '2026-05-25', '2026-07-14', '2026-08-15',
    '2026-11-01', '2026-11-11', '2026-12-25'
]

def generate_future_calendar(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    df_future = pd.DataFrame({'date': dates})
    df_future['month'] = df_future['date'].dt.month
    df_future['day_of_week'] = df_future['date'].dt.dayofweek
    df_future['year'] = df_future['date'].dt.year
    df_future['week_end'] = df_future['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_future['ferie'] = df_future['date'].dt.strftime('%Y-%m-%d').isin(FERIE_2026).astype(int)
    return df_future

def run_simulation():
    print("Loading historical data and fitting SARIMAX model...")
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    model = evaluate_model(df_hist)

    last_date = df_hist['date'].max()
    future_df = generate_future_calendar(last_date + pd.Timedelta(days=1), '2026-12-31')

    # Scenario A: Service always available (Disponibilite_de_service = 0)
    future_A = future_df[['ferie', 'week_end']].copy()
    future_A['Disponibilite_de_service'] = 0.0
    pred_available = np.maximum(0, model.predict(future_A))

    # Scenario B: Service unavailable every day (Disponibilite_de_service = 1)
    future_B = future_df[['ferie', 'week_end']].copy()
    future_B['Disponibilite_de_service'] = 1.0
    pred_unavailable = np.maximum(0, model.predict(future_B))

    output_df = pd.DataFrame({
        'date': future_df['date'].dt.strftime('%Y-%m-%d'),
        'ferie': future_df['ferie'],
        'week_end': future_df['week_end'],
        'prediction_available': pred_available,
        'prediction_unavailable': pred_unavailable
    })

    output_path = 'data/processed/Forecast_Scenarios.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Forecast scenarios exported side by side to {output_path}")

    print("\nFirst 10 rows of forecast output:")
    print(output_df.head(10).to_string(index=False))

    return output_df

if __name__ == '__main__':
    run_simulation()

