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

CHANNELS = ['digital', 'bureau']

def generate_future_calendar(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    df_future = pd.DataFrame({'date': dates})
    df_future['month'] = df_future['date'].dt.month
    df_future['day_of_week'] = df_future['date'].dt.dayofweek
    df_future['year'] = df_future['date'].dt.year
    df_future['week_end'] = df_future['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2026-2027 French Holidays
    holidays = pd.to_datetime([
        '2026-01-01', '2026-04-06', '2026-05-01', '2026-05-08', '2026-05-14', '2026-05-25',
        '2026-07-14', '2026-08-15', '2026-11-01', '2026-11-11', '2026-12-25',
        '2027-01-01', '2027-03-29', '2027-05-01', '2027-05-06', '2027-05-17',
        '2027-07-14', '2027-08-15', '2027-11-01', '2027-11-11', '2027-12-25'
    ])
    df_future['ferie'] = df_future['date'].isin(holidays).astype(int)
    return df_future

def run_simulation():
    print("Loading historical data and fitting per-channel SARIMAX models with Fourier annual seasonality...")
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])

    last_date = df_hist['date'].max()
    future_end = pd.to_datetime('2027-08-31')
    future_df = generate_future_calendar(last_date + pd.Timedelta(days=1), future_end)

    daily_output = pd.DataFrame({
        'date': future_df['date'].dt.strftime('%Y-%m-%d'),
        'ferie': future_df['ferie'],
        'week_end': future_df['week_end'],
    })

    future_base = future_df[['date', 'ferie', 'week_end']].copy()
    future_base['bug_technique'] = 0.0

    # Fit per-channel models
    for canal in CHANNELS:
        print(f"Fitting SARIMAX for channel '{canal}'...")
        model_canal = evaluate_model(df_hist, canal_verification=canal)

        future_A = future_base.copy()
        future_A['Disponibilite_de_service'] = 0.0
        pred_A = np.maximum(0, model_canal.predict(future_A))

        future_B = future_base.copy()
        future_B['Disponibilite_de_service'] = 1.0
        pred_B = np.maximum(0, model_canal.predict(future_B))

        daily_output[f'prediction_available_{canal}'] = np.round(pred_A, 2)
        daily_output[f'prediction_unavailable_{canal}'] = np.round(pred_B, 2)
        future_df[f'pred_available_{canal}'] = pred_A
        future_df[f'pred_unavailable_{canal}'] = pred_B

    # Sum total predictions across channels
    daily_output['prediction_available'] = sum(daily_output[f'prediction_available_{canal}'] for canal in CHANNELS)
    daily_output['prediction_unavailable'] = sum(daily_output[f'prediction_unavailable_{canal}'] for canal in CHANNELS)
    future_df['pred_available'] = sum(future_df[f'pred_available_{canal}'] for canal in CHANNELS)
    future_df['pred_unavailable'] = sum(future_df[f'pred_unavailable_{canal}'] for canal in CHANNELS)

    daily_output_path = 'data/processed/Forecast_Scenarios.csv'
    daily_output.to_csv(daily_output_path, index=False)
    print(f"Daily forecast scenarios exported to {daily_output_path}")

    # Monthly Aggregation for Executive Dashboard (Tableau)
    future_df['is_working_day'] = ((future_df['week_end'] == 0) & (future_df['ferie'] == 0)).astype(int)

    agg_dict = {
        'ferie': 'sum',
        'is_working_day': 'sum',
        'pred_available': 'sum',
        'pred_unavailable': 'sum'
    }
    for canal in CHANNELS:
        agg_dict[f'pred_available_{canal}'] = 'sum'
        agg_dict[f'pred_unavailable_{canal}'] = 'sum'

    monthly_output = future_df.groupby(pd.Grouper(key='date', freq='MS')).agg(agg_dict).reset_index()

    rename_dict = {
        'pred_available': 'prediction_available',
        'pred_unavailable': 'prediction_unavailable',
        'is_working_day': 'working_days'
    }
    for canal in CHANNELS:
        rename_dict[f'pred_available_{canal}'] = f'prediction_available_{canal}'
        rename_dict[f'pred_unavailable_{canal}'] = f'prediction_unavailable_{canal}'

    monthly_output.rename(columns=rename_dict, inplace=True)
    monthly_output['date'] = monthly_output['date'].dt.strftime('%Y-%m')

    # Round integer volumes
    for col in monthly_output.columns:
        if 'prediction' in col:
            monthly_output[col] = np.round(monthly_output[col])

    monthly_output_path = 'data/processed/monthly_forecast_scenarios.csv'
    monthly_output.to_csv(monthly_output_path, index=False)
    print(f"Monthly forecast scenarios exported to {monthly_output_path}")

    print("\n--- Monthly Forecast Scenarios (Per Channel & Total) ---")
    print(monthly_output.to_string(index=False))

    return monthly_output

if __name__ == '__main__':
    run_simulation()


