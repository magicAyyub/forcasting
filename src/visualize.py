import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

FORECAST_START = pd.to_datetime('2026-09-01')
CHANNELS = ['digital', 'bureau']

def _plot_channel_scenarios(ax, hist_series, fcst_df, avail_col, unavail_col, title):
    ax.plot(hist_series.index, hist_series.values, label='Historique', marker='o', color='#1f77b4', linewidth=2)
    ax.plot(fcst_df.index, fcst_df[avail_col], label='Scénario A (Disponible)', marker='^', color='#2ca02c', linewidth=2.2, linestyle='--')
    ax.plot(fcst_df.index, fcst_df[unavail_col], label='Scénario B (Indisponible)', marker='v', color='#d62728', linewidth=2.2, linestyle=':')
    ax.axvline(x=FORECAST_START, color='gray', linestyle='--', alpha=0.7, label='Début Prévision (Sept. 2026)')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume', fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

def create_visualizations():
    df_hist = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df_hist['date'] = pd.to_datetime(df_hist['date'])

    df_fcst = pd.read_csv('data/processed/Forecast_Scenarios.csv')
    df_fcst['date'] = pd.to_datetime(df_fcst['date'])

    # Monthly resampling
    hist_total_monthly = df_hist.groupby('date')['t_verified_canal_sum'].sum().resample('MS').sum()
    fcst_total_monthly = df_fcst.set_index('date').resample('MS')[['prediction_available', 'prediction_unavailable']].sum()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Subplot 1: Total Volume
    _plot_channel_scenarios(
        axes[0], hist_total_monthly, fcst_total_monthly,
        'prediction_available', 'prediction_unavailable',
        'Volume Mensuel Total (Tous Canaux) : Historique & Prévisions (SARIMAX)'
    )

    # Subplot 2 & 3: Digital & Bureau
    for i, canal in enumerate(CHANNELS, start=1):
        hist_canal_monthly = df_hist[df_hist['canal_verification'] == canal].set_index('date')['t_verified_canal_sum'].resample('MS').sum()
        fcst_canal_monthly = df_fcst.set_index('date').resample('MS')[[f'prediction_available_{canal}', f'prediction_unavailable_{canal}']].sum()

        _plot_channel_scenarios(
            axes[i], hist_canal_monthly, fcst_canal_monthly,
            f'prediction_available_{canal}', f'prediction_unavailable_{canal}',
            f"Volume Mensuel - Canal '{canal.capitalize()}' : Historique & Prévisions (SARIMAX)"
        )

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel('Mois', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig('visualizations/Forecast_Volumes.png', dpi=300)
    print("Visualization saved as visualizations/Forecast_Volumes.png")

if __name__ == '__main__':
    create_visualizations()


