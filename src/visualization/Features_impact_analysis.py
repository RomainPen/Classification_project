import pandas as pd
import shap
import matplotlib.pyplot as plt


class FeaturesImpactAnalysis:
    def __init__(self, model, x_df : pd.DataFrame):
        self.model = model
        self.x_df=x_df
        
    
    def Features_importance(self) -> pd.DataFrame :
        """
        Calculate and return feature importance scores as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing feature names and their importance scores in percentage.
        """
        df_features_importance = (pd.DataFrame({'Features': self.model.feature_names_in_,
              'Features importance (in %)': (self.model.feature_importances_)*100}))
        return df_features_importance.sort_values(by='Features importance (in %)', ascending=False)
    
    
    def SHAP_explainer(self, file_saving) :
        """
        Create and save a SHAP summary plot for model interpretability.

        Args:
            file_saving: The file path for saving the SHAP summary plot in PDF format.

        Returns:
            None
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_df)
        shap.summary_plot(shap_values, self.x_df, show=False) #shap_summary_plot = 
        plt.savefig(file_saving, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close()