import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    # Load historical clean
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    monthly_hist = df_hist.groupby([pd.Grouper(key='date', freq='ME'), 'canal_verification'])['t_verified_canal_sum'].sum().unstack(fill_value=0)
    
    # Load forecasts
    df_fcst = pd.read_csv('data/processed/Forecast_Scenarios.csv')
    df_fcst['date'] = pd.to_datetime(df_fcst['date'])
    
    # Let's plot Scenario A (Optimal)
    opt_fcst = df_fcst[df_fcst['scenario'] == 'Scenario A (Optimal: No Outages)']
    monthly_opt = opt_fcst.groupby(pd.Grouper(key='date', freq='ME'))[['digital_volume', 'bureau_volume']].sum()
    
    # Combine history and forecast for plotting
    digital_series = pd.concat([monthly_hist['digital'], monthly_opt['digital_volume']])
    bureau_series = pd.concat([monthly_hist['bureau'], monthly_opt['bureau_volume']])
    
    # Plot 1: Volume over time
    plt.figure(figsize=(12, 6))
    plt.plot(digital_series.index, digital_series.values, label='Digital Volume', marker='o')
    plt.plot(bureau_series.index, bureau_series.values, label='Bureau Volume', marker='s')
    
    # Vertical line to show forecast start
    forecast_start = pd.to_datetime('2026-06-24')
    plt.axvline(x=forecast_start, color='red', linestyle='--', label='Forecast Start (June 2026)')
    
    plt.title('Monthly Verification Volumes: Historical & Forecasted (Optimal Scenario)')
    plt.xlabel('Date')
    plt.ylabel('Total Monthly Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/Forecast_Volumes.png')
    
    # Plot 2: Digital Ratio over time
    ratio_series = digital_series / (digital_series + bureau_series) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(ratio_series.index, ratio_series.values, label='Digital %', color='green', marker='^')
    plt.axvline(x=forecast_start, color='red', linestyle='--', label='Forecast Start')
    
    # Highlight transition
    transition_start = pd.to_datetime('2026-10-01')
    plt.axvspan(transition_start, ratio_series.index[-1], color='yellow', alpha=0.2, label='Target Ratio Transition Period')
    
    plt.title('Digital Verification Percentage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Digital %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/Forecast_Ratio.png')
    
    print("Visualizations saved as visualizations/Forecast_Volumes.png and visualizations/Forecast_Ratio.png")

if __name__ == '__main__':
    create_visualizations()
