import pandas as pd
import shap
import matplotlib.pyplot as plt


class FeaturesImpactAnalysis:
    def __init__(self, model, x_df):
        self.model = model
        self.x_df=x_df
        
    
    def Features_importance(self):
        df_features_importance = (pd.DataFrame({'Features': self.model.feature_names_in_,
              'Features importance (in %)': (self.model.feature_importances_)*100}))
        return df_features_importance.sort_values(by='Features importance (in %)', ascending=False)
    
    
    def SHAP_explainer(self, file_saving) :
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_df)
        shap.summary_plot(shap_values, self.x_df, show=False) #shap_summary_plot = 
        plt.savefig(file_saving, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close()