import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

# Import your custom data loader modules
from EnergyDataLoader.energydataloader import EnergyDataLoader
from EnergySQL.energysql import EnergySQL

# Import refactored modules
from data_processor import (combine_ecmwf_pkl_files, resample, get_true_wind_power, 
                            create_ecmwf_weather_matrices, prepare_train_data_3d, prepare_pred_data_3d)
from train import train_3d_cnn

def _interpolate_predictions(results):
    preds = results['predictions']
    if len(preds) < 2: 
        return results
        
    # Stretch using cubic interpolation to 96 points (15-minute intervals)
    cubic_interp = interp1d(
        np.arange(len(preds)), preds, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )
    results['predictions'] = cubic_interp(np.linspace(0, len(preds) - 1, 96))
    return results

def predict_wind_power(start_date, end_date, n=2, period=2, lookback=35, epochs=20, batch_size=64, lr=0.005):
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    results_list = []

    for pred_date in prediction_dates:
        print(f"\n{'='*50}\nTraining for prediction date: {pred_date.strftime('%Y%m%d')}\n{'='*50}")
        
        start_train_str = (pred_date - pd.Timedelta(days=lookback)).strftime('%Y%m%d')
        end_train_str = (pred_date - pd.Timedelta(days=2)).strftime('%Y%m%d')

        # 1. Fetch data
        df, pred = combine_ecmwf_pkl_files(pred_date.strftime('%Y%m%d'), start_train_str, end_train_str, n)
        df_south, df_north = resample(df)
        pred_south, pred_north = resample(pred)
        y = get_true_wind_power(start_train_str, end_train_str)
        
        # 2. Convert to matrices
        gm_south = create_ecmwf_weather_matrices(df_south)
        gm_north = create_ecmwf_weather_matrices(df_north)
        gm_south_pred = create_ecmwf_weather_matrices(pred_south)
        gm_north_pred = create_ecmwf_weather_matrices(pred_north)

        # 3. Prepare Tensor sequences
        X_s, y_s = prepare_train_data_3d(gm_south, y, lookback=period)
        X_n, y_n = prepare_train_data_3d(gm_north, y, lookback=period)
        
        X_pred_s = prepare_pred_data_3d(gm_south_pred, lookback=period)
        X_pred_n = prepare_pred_data_3d(gm_north_pred, lookback=period)

        if X_s is None or X_n is None:
            print("Data incomplete for this date, skipping...")
            continue
            
        # Ensure south and north datasets have the same length (defensive programming)
        min_len = min(len(X_s), len(X_n))
        X_s, X_n, y_val = X_s[:min_len], X_n[:min_len], y_s[:min_len]

        # 4. Train model
        results = train_3d_cnn(X_s, X_n, y_val, X_pred_s, X_pred_n, epochs=epochs, batch_size=batch_size, lr=lr)
        
        # 5. Interpolate
        results = _interpolate_predictions(results)
        results_list.append(results)

    # Aggregate results into a DataFrame
    all_values = [val for res in results_list for val in res['predictions'].flatten()]
    timestamps = pd.date_range(start=prediction_dates[0] + pd.Timedelta(days=n), periods=len(all_values), freq='15min')
    
    final_df = pd.DataFrame({f'wind_n1_N{n}': all_values}, index=timestamps)
    final_df.index.name = 'timestamp'
    final_df = final_df.rolling(window=1, min_periods=1, center=True).mean()
    final_df = np.clip(final_df, 1000, 20000)
    
    return final_df

if __name__ == "__main__":
    date = '20251001'
    end = '20251016'
    
    # Create directory for saving results
    output_dir = 'Experiment/windpower_forecast'
    os.makedirs(output_dir, exist_ok=True)
    
    results = predict_wind_power(start_date=date, end_date=end, n=2)
    # output_path = os.path.join(output_dir, 'wind_n1_N.csv')
    # results.to_csv(output_path, index=True)
    
    # print(f"Prediction completed! Results saved to {output_path}")
