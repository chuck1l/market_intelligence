from data_clean import DataClean

if __name__ == '__main__':
    location = '../data/spy_1h_ext_hours.csv'
    # Clean the data, set up the columns to predict on
    spy = DataClean(location)
    spy_clean = spy.data_prepared()
    # Time for EDA
    print(spy_clean.head())