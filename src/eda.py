import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor

class StockEda(object):
    def __init__(self, X_train, y_train, g_label):
        self.X_train = X_train
        self.y_train = y_train
        self.g_label = g_label # setting the label for the graph and save location
    def feature_importance(self):
        save_location = '../img/' + self.g_label + 'feature_importance.png'
        labels = pd.Series(self.X_train.columns, name='features')
        graph_labels = list(self.X_train.columns)
        # Using Random Forest Model with default inputs
        randomforest_model = RandomForestRegressor()
        randomforest_model.fit(self.X_train, self.y_train)
        # Get feature importances
        importances = pd.Series(randomforest_model.feature_importances_, name='Feature Importance')
        features_df = pd.concat([labels, importances], axis=1)
        features_df = features_df.sort_values(by='Feature Importance', ascending=False)
        x = np.arange(features_df.shape[0]) # label locations
        width = 0.35 # setting the width of each bar for the labels
        # Plotting the Random Forest Model Feature importance
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


if __name__ == '__main__':
    pass 