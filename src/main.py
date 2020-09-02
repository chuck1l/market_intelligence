from data_clean_1h import DataCleanOneHour
from data_clean_1d import DataCleanOneDay

if __name__ == '__main__':
    location_1h = '../data/spy_1h_ext_hours.csv'
    location_1d = '../data/spy_1d.csv'
    # Clean the data, set up the columns to predict on
    spy_1h = DataCleanOneHour(location_1h)
    spy_1h_clean = spy_1h.data_prepared() # 1 hour data for predicting time of day high/low

    spy_1d = DataCleanOneDay(location_1d)
    spy_1d_clean = spy_1d.data_prepared()
    print(spy_1d_clean.head())




    
