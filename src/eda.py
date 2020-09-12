import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class StockEda(object):
    def __init__(self, data, g_label):
        self.g_label = g_label
        self.df = data
        mask = (self.df['year']) >= 2000 & (self.df['year'] < 2016)
        self.df = data[mask]

    def ensure_float(self):
        self.df.drop('date', axis=1, inplace=True)
        cols = self.df.columns
        self.df[cols] = self.df[cols].astype(float).round(2)

    def normalize_features(self):
        cols = self.df.columns
        scaler = StandardScaler()
        self.scaled = self.df.copy()
        self.scaled[cols] = scaler.fit_transform(self.scaled[cols])

    def create_X_y_train(self):
        self.y_train = self.scaled[self.g_label]
        self.X_train = self.scaled.copy()
        self.X_train.drop(columns=['tomorrow_high', 'tomorrow_low', 'ticks'], axis=1, inplace=True)

    def feature_importance(self):
        save_location = '../imgs/' + self.g_label + '_feature_importance.png'
        labels = pd.Series(self.X_train.columns, name='features')
        # Using Random Forest Model with default inputs
        randomforest_model = RandomForestRegressor()
        randomforest_model.fit(self.X_train, self.y_train)
        # Get feature importances
        importances = pd.Series(randomforest_model.feature_importances_, name='Feature Importance')
        features_df = pd.concat([labels, importances], axis=1)
        features_df = features_df.sort_values(by='Feature Importance', ascending=False)
        graph_labels = list(features_df.features)
        x = np.arange(features_df.shape[0]) # label locations
        width = 0.35 # setting the width of each bar for the labels
        # Plotting the Random Forest Model Feature importance
        toggle = False
        if toggle:
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.bar(x, features_df['Feature Importance'], width, label=self.g_label)
            ax.set_ylabel('Importance Score')
            ax.set_title('Feature Importance As Per Random Forest')
            ax.set_xticks(x)
            ax.set_xticklabels(graph_labels, rotation='vertical')
            ax.legend()
            fig.tight_layout()
            plt.savefig(save_location)
            plt.show()
        self.result = list(features_df['features'][:5].values)
        high_low_lst = ['high', 'low']
        for item in high_low_lst:
            if item not in self.result:
                self.result.append(item)
        
    def get_principal_comp(self):
        self.ensure_float()
        self.normalize_features()
        self.create_X_y_train()
        self.feature_importance()
        return self.result
       
if __name__ == '__main__':
    pass