# Standard Imports
import pandas as pd
import numpy as np
from datetime import date 
from datetime import timedelta 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
from sklearn.metrics import mean_squared_error
# Source Data From Yahoo

result_df = pd.read_csv('../data/30_day_complete_result_df.csv')
result_df['Date']= pd.to_datetime(result_df['Date'])
result_df.set_index('Date', inplace=True)

rmse_high_persis = np.sqrt(mean_squared_error(result_df['persis_high'].values, result_df['tmr_high'].values))
rmse_low_persis = np.sqrt(mean_squared_error(result_df['persis_low'].values, result_df['tmr_low'].values))
rmse_high_pred = np.sqrt(mean_squared_error(result_df['predicted_high'].values, result_df['tmr_high'].values))
rmse_low_pred = np.sqrt(mean_squared_error(result_df['predicted_low'].values, result_df['tmr_low'].values))
print(f'Persistence Model RMSE High of Day: {rmse_high_persis:0.3f}')
print(f'Persistence Model RMSE Low of Day: {rmse_low_persis:0.3f}')
print(f'Predicted Model RMSE High of Day: {rmse_high_pred:0.3f}')
print(f'Predicted Model RMSE Low of Day: {rmse_low_pred:0.3f}')
print(result_df.head())

toggle = True
if toggle:
    # Plot the true vs persistence, high
    plt.figure(figsize=(22, 12))
    plt.plot(result_df.index, result_df['persis_high'], c='r', marker='o', markersize=1, label='Prediction')
    plt.plot(result_df.index, result_df['tmr_high'], 'k--', label='True Value')
    plt.ylabel('Price ($)', fontsize=20, c='k')
    plt.xlabel('Date', fontsize=20, c='k')
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', colors='k', labelsize=20)
    plt.tick_params(axis='y', colors='k', labelsize=20)
    plt.title('Predictions vs. True High Price (Baseline Model)', fontsize=24)
    plt.legend()
    plt.tight_layout
    plt.savefig('../imgs/basline_high_results.png')
    plt.show();
    # Low
    plt.figure(figsize=(22, 12))
    plt.plot(result_df.index, result_df['persis_low'], c='r', marker='o', markersize=1, label='Prediction')
    plt.plot(result_df.index, result_df['tmr_low'], 'k--', label='True Value')
    plt.ylabel('Price ($)', fontsize=20, c='k')
    plt.xlabel('Date', fontsize=20, c='k')
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', colors='k', labelsize=20)
    plt.tick_params(axis='y', colors='k', labelsize=20)
    plt.title('Predictions vs. True Low Price (Baseline Model)', fontsize=24)
    plt.legend()
    plt.tight_layout
    plt.savefig('../imgs/baseline_low_results.png')
    plt.show();

    # Plot the true vs predictions, high
    plt.figure(figsize=(22, 12))
    plt.plot(result_df.index, result_df['predicted_high'], c='r', marker='o', markersize=1, label='Prediction')
    plt.plot(result_df.index, result_df['tmr_high'], 'k--', label='True Value')
    plt.ylabel('Price ($)', fontsize=20, c='k')
    plt.xlabel('Date', fontsize=20, c='k')
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', colors='k', labelsize=20)
    plt.tick_params(axis='y', colors='k', labelsize=20)
    plt.title('Predictions vs. True High Price (Random Forest Model)', fontsize=24)
    plt.legend()
    plt.tight_layout
    plt.savefig('../imgs/randomforest_high_results.png')
    plt.show();
    # Low
    plt.figure(figsize=(22, 12))
    plt.plot(result_df.index, result_df['predicted_low'], c='r', marker='o', markersize=1, label='Prediction')
    plt.plot(result_df.index, result_df['tmr_low'], 'k--', label='True Value')
    plt.ylabel('Price ($)', fontsize=20, c='k')
    plt.xlabel('Date', fontsize=20, c='k')
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', colors='k', labelsize=20)
    plt.tick_params(axis='y', colors='k', labelsize=20)
    plt.title('Predictions vs. True Low Price (Random Forest Model)', fontsize=24)
    plt.legend()
    plt.tight_layout
    plt.savefig('../imgs/randomforest_low_results.png')
    plt.show();