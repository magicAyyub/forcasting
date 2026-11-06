import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from prophet import Prophet
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

def prepare_total_volume_data(df):
    daily_total = df.groupby('date').agg({
        't_verified_canal_sum': 'sum',
        'ferie': 'first',
        'week_end': 'first',
        'Disponibilite_de_service': 'first',
        'bug_technique': 'max',
        'month': 'first',
        'day_of_week': 'first'
    }).reset_index()
    daily_total = daily_total.sort_values('date').reset_index(drop=True)
    return daily_total

class MLWrapper:
    def __init__(self, model_type='ridge'):
        self.model_type = model_type
        cat_features = ['month', 'day_of_week']
        num_features = ['ferie', 'week_end', 'Disponibilite_de_service', 'bug_technique']
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', StandardScaler(), num_features)
        ])
        
        if model_type == 'ridge':
            reg = Ridge(alpha=1.0)
        elif model_type == 'rf':
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            reg = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            
        self.model = Pipeline([('preprocessor', preprocessor), ('regressor', reg)])
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class ProphetWrapper:
    def __init__(self):
        self.model = None
        
    def fit(self, X, y):
        self.model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        self.model.add_regressor('ferie')
        self.model.add_regressor('Disponibilite_de_service')
        self.model.add_regressor('bug_technique')
        df_p = X.copy()
        df_p['ds'] = df_p['date']
        df_p['y'] = y
        self.model.fit(df_p)
        
    def predict(self, X):
        df_p = X.copy()
        df_p['ds'] = df_p['date']
        preds = self.model.predict(df_p)
        return preds['yhat'].values

class SARIMAXWrapper:
    def __init__(self):
        self.model_res = None
        
    def fit(self, X, y):
        exog = X[['ferie', 'week_end', 'Disponibilite_de_service', 'bug_technique']]
        # simple SARIMAX (1, 1, 1) x (1, 0, 1, 7)
        self.model = sm.tsa.SARIMAX(y.values, exog=exog.values, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        self.model_res = self.model.fit(disp=False)

    def predict(self, X):
        exog = X[['ferie', 'week_end', 'Disponibilite_de_service', 'bug_technique']]
        return self.model_res.forecast(steps=len(X), exog=exog.values)

def run_benchmark():
    df = pd.read_csv('data/processed/Acquisition_LIN_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    daily_df = prepare_total_volume_data(df)
    
    # We will use the last 5 periods of 30 days for testing (150 days total for CV testing)
    tscv = TimeSeriesSplit(n_splits=5, test_size=30)
    
    X = daily_df[['date', 'month', 'day_of_week', 'ferie', 'week_end', 'Disponibilite_de_service', 'bug_technique']]
    y = daily_df['t_verified_canal_sum']
    
    models = {
        'Ridge Regression': MLWrapper('ridge'),
        'Random Forest': MLWrapper('rf'),
        'LightGBM': MLWrapper('lightgbm'),
        'Prophet': ProphetWrapper(),
        'SARIMAX': SARIMAXWrapper()
    }
    
    results = []
    
    for name, model in models.items():
        maes, mses, mapes = [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit & Predict
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            maes.append(mean_absolute_error(y_test, preds))
            mses.append(root_mean_squared_error(y_test, preds))
            # MAPE can explode if y_test is near 0, so we use a safe calculation
            safe_mape = np.mean(np.abs((y_test.values - preds) / np.maximum(y_test.values, 1))) * 100
            mapes.append(safe_mape)
            
        results.append({
            'Model': name,
            'Avg_MAE': np.mean(maes),
            'Avg_RMSE': np.mean(mses),
            'Avg_MAPE': np.mean(mapes)
        })
        
    res_df = pd.DataFrame(results).sort_values('Avg_MAE')
    print("--- Model Selection Leaderboard ---")
    print(res_df.to_string(index=False, float_format="{:.2f}".format))
    return res_df

if __name__ == '__main__':
    run_benchmark()
