import pandas as pd
import numpy as np
import statsmodels.api as sm

class SARIMAXWrapper:
    def __init__(self):
        self.model_res = None
        
    def fit(self, X, y):
        exog = X[['ferie', 'week_end', 'Disponibilite_de_service']]
        # simple SARIMAX (1, 1, 1) x (1, 0, 1, 7)
        self.model = sm.tsa.SARIMAX(y.values, exog=exog.values, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        self.model_res = self.model.fit(disp=False)
        return self
        
    def predict(self, X):
        exog = X[['ferie', 'week_end', 'Disponibilite_de_service']]
        return self.model_res.forecast(steps=len(X), exog=exog.values)

def prepare_data_for_total_volume(df):
    daily_total = df.groupby('date').agg({
        't_verified_canal_sum': 'sum',
        'ferie': 'first',
        'week_end': 'first',
        'Disponibilite_de_service': 'first'
    }).reset_index()
    return daily_total

def evaluate_model(df):
    daily_total = prepare_data_for_total_volume(df)
    daily_total = daily_total.sort_values('date').reset_index(drop=True)
    
    X = daily_total[['date', 'ferie', 'week_end', 'Disponibilite_de_service']]
    y = daily_total['t_verified_canal_sum']
    
    # Fit SARIMAX on all historical data
    final_model = SARIMAXWrapper().fit(X, y)
    return final_model

if __name__ == '__main__':
    df = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    model = evaluate_model(df)
    print("\nSARIMAX model trained successfully on all historical data.")
