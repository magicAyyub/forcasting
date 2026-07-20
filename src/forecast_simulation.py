import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
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

def apply_ratio(df, scenario_name):
    """ Applies the channel split ratios based on the implementation plan mathematics. """
    results = []
    for _, row in df.iterrows():
        # Baseline ratios for weekend/holidays
        if row['ferie'] == 1:
            digital_ratio = 0.9941
        elif row['week_end'] == 1:
            digital_ratio = 0.8639
        else:
            # Weekday ratio transitions
            if row['month'] <= 10:  # Up to October 2026 (Overall 75% target)
                digital_ratio = 0.7248
            elif row['month'] == 11: # November 2026 (Overall 77% target)
                digital_ratio = 0.7489
            else: # December 2026 (Overall 80% target)
                digital_ratio = 0.7849
                
        bureau_ratio = 1.0 - digital_ratio
        
        results.append({
            'date': row['date'],
            'scenario': scenario_name,
            'total_volume': row['predicted_total'],
            'digital_volume': row['predicted_total'] * digital_ratio,
            'bureau_volume': row['predicted_total'] * bureau_ratio,
            'digital_ratio_applied': digital_ratio
        })
    return pd.DataFrame(results)

def run_simulation():
    # 1. Load data and train model
    print("Loading data and training model on historical data...")
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    model = evaluate_model(df_hist)
    
    # 2. Generate future dates
    last_date = df_hist['date'].max()
    future_df = generate_future_calendar(last_date + pd.Timedelta(days=1), '2026-12-31')
    
    # 3. Simulate Scenarios
    # Scenario A: Optimal (0.0 outages)
    future_A = future_df.copy()
    future_A['Disponibilite_de_service'] = 0.0
    future_A['predicted_total'] = model.predict(future_A)
    res_A = apply_ratio(future_A, 'Scenario A (Optimal: No Outages)')
    
    # Scenario B: Standard (Outages happen 8% of weekdays)
    np.random.seed(42)
    future_B = future_df.copy()
    is_weekday = (future_B['week_end'] == 0) & (future_B['ferie'] == 0)
    future_B['Disponibilite_de_service'] = 0.0
    # Randomly assign 1.0 to 8% of weekdays
    future_B.loc[is_weekday, 'Disponibilite_de_service'] = np.random.choice([0.0, 1.0], size=is_weekday.sum(), p=[0.92, 0.08])
    future_B['predicted_total'] = model.predict(future_B)
    res_B = apply_ratio(future_B, 'Scenario B (Standard: Historical Outage Rate)')
    
    # 4. Combine and save
    all_forecasts = pd.concat([res_A, res_B], ignore_index=True)
    all_forecasts.to_csv('data/processed/Forecast_Scenarios.csv', index=False)
    print("\nForecasts generated and saved to data/processed/Forecast_Scenarios.csv")
    
    # Print some summaries
    print("\n--- Summary of Scenario A (Optimal) ---")
    monthly_A = res_A[res_A['scenario'] == 'Scenario A (Optimal: No Outages)'].set_index('date').resample('ME').sum()
    monthly_A['digital_pct'] = monthly_A['digital_volume'] / monthly_A['total_volume'] * 100
    print(monthly_A[['total_volume', 'digital_volume', 'bureau_volume', 'digital_pct']])

if __name__ == '__main__':
    run_simulation()
