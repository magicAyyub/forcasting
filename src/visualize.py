import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    # Load historical clean data
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    daily_hist = df_hist.groupby('date')['t_verified_canal_sum'].sum().reset_index()
    monthly_hist = daily_hist.set_index('date').resample('ME')['t_verified_canal_sum'].sum()

    # Load forecast scenarios
    df_fcst = pd.read_csv('data/processed/Forecast_Scenarios.csv')
    df_fcst['date'] = pd.to_datetime(df_fcst['date'])
    monthly_fcst = df_fcst.set_index('date').resample('ME')[['prediction_available', 'prediction_unavailable']].sum()

    # Plot Monthly Total Verification Volumes
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_hist.index, monthly_hist.values, label='Historical Volume', marker='o', color='blue')
    plt.plot(monthly_fcst.index, monthly_fcst['prediction_available'], label='Forecast - Service Available (Scenario A)', marker='^', color='green')
    plt.plot(monthly_fcst.index, monthly_fcst['prediction_unavailable'], label='Forecast - Service Unavailable (Scenario B)', marker='v', color='red')

    forecast_start = pd.to_datetime('2026-06-24')
    plt.axvline(x=forecast_start, color='gray', linestyle='--', label='Forecast Start (June 2026)')

    plt.title('Monthly Total Verification Volume: Historical & Forecasting Scenarios (SARIMAX)')
    plt.xlabel('Date')
    plt.ylabel('Total Verification Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/Forecast_Volumes.png')

    print("Visualization saved as visualizations/Forecast_Volumes.png")

if __name__ == '__main__':
    create_visualizations()

