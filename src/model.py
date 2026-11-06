import pandas as pd
import numpy as np
import statsmodels.api as sm

# Exogenous variables + annual Fourier terms to capture macro monthly seasonality
BASE_EXOG_COLS = ['ferie', 'week_end', 'Disponibilite_de_service', 'bug_technique']
FOURIER_COLS = ['sin_1', 'cos_1', 'sin_2', 'cos_2']
EXOG_COLS = BASE_EXOG_COLS + FOURIER_COLS

def add_fourier_terms(df):
    """Add annual Fourier terms to capture macro monthly seasonality (August slump, September rentree)."""
    df = df.copy()
    if 'date' in df.columns:
        doy = pd.to_datetime(df['date']).dt.dayofyear
    else:
        doy = df.index.dayofyear
    df['sin_1'] = np.sin(2 * np.pi * doy / 365.25)
    df['cos_1'] = np.cos(2 * np.pi * doy / 365.25)
    df['sin_2'] = np.sin(4 * np.pi * doy / 365.25)
    df['cos_2'] = np.cos(4 * np.pi * doy / 365.25)
    return df

class SARIMAXWrapper:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res = None

    def fit(self, X, y):
        X_feat = add_fourier_terms(X)
        exog = X_feat[EXOG_COLS]

        self.model = sm.tsa.SARIMAX(
            y.values,
            exog=exog.values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = self.model.fit(disp=False)
        return self

    def predict(self, X):
        X_feat = add_fourier_terms(X)
        exog = X_feat[EXOG_COLS]
        return self.model_res.forecast(steps=len(X), exog=exog.values)

def prepare_data_for_total_volume(df):
    daily_df = df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_total = daily_df.groupby('date').agg({
        't_verified_canal_sum': 'sum',
        'ferie': 'first',
        'week_end': 'first',
        'Disponibilite_de_service': 'first',
        'bug_technique': 'max'
    }).reset_index()
    daily_total = add_fourier_terms(daily_total)
    return daily_total

def prepare_data_for_channel(df, canal_verification):
    daily_df = df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_channel = daily_df[daily_df['canal_verification'] == canal_verification].copy()
    daily_channel = daily_channel.sort_values('date').reset_index(drop=True)
    daily_channel = add_fourier_terms(daily_channel)
    return daily_channel

def evaluate_model(df, canal_verification=None):
    """Fit SARIMAX on a single channel if canal_verification is provided, otherwise on total volume."""
    if canal_verification is not None:
        daily_data = prepare_data_for_channel(df, canal_verification)
    else:
        daily_data = prepare_data_for_total_volume(df)
    daily_data = daily_data.sort_values('date').reset_index(drop=True)

    X = daily_data[['date'] + BASE_EXOG_COLS]
    y = daily_data['t_verified_canal_sum']

    final_model = SARIMAXWrapper().fit(X, y)
    return final_model

if __name__ == '__main__':
    df = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    model_total = evaluate_model(df)
    model_digital = evaluate_model(df, canal_verification='digital')
    model_bureau = evaluate_model(df, canal_verification='bureau')
    print("\nSARIMAX models (Total, Digital, Bureau) trained successfully.")


