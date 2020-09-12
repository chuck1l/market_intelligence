from data_clean_1h import DataCleanOneHour
from data_clean_1d import DataCleanOneDay
from eda import StockEda
from price_predict import PredictPrice

if __name__ == '__main__':
    location_1h = '../data/spy_1h_ext_hours.csv'
    location_1d = '../data/spy_1d.csv'
    # Clean the data, set up the columns to predict time @ high/low on
    spy_1h = DataCleanOneHour(location_1h)
    spy_1h_clean = spy_1h.data_prepared() # 1 hour data for predicting time of day high/low
    # Clean the data set up the columns to predict price on
    spy_1d = DataCleanOneDay(location_1d)
    spy_1d_clean = spy_1d.data_prepared()
    # Select the columns to keep for high/low predictions PCA, Random Forest
    # for tomorrow low analysis 
    spy_low = StockEda(spy_1d_clean, 'tomorrow_low')
    low_features = spy_low.get_principal_comp()
    # for tomorrow high analysis
    spy_high = StockEda(spy_1d_clean, 'tomorrow_high')
    high_features = spy_high.get_principal_comp()
    # Add the target column back to the datasets
    high_features.append('tomorrow_high')
    low_features.append('tomorrow_low')
    # Create the dataframe based on feature importance
    high_stock = spy_1d_clean[high_features].copy()
    low_stock = spy_1d_clean[low_features].copy()

    high_stock.to_csv('../data/lstm_testdata.csv')
    print(high_stock.head())
    # PredictPrice(high_stock, 'tomorrow_high').prediction()

    
    




    
