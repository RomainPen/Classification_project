import shap
import matplotlib.pyplot as plt

def compute_features_importance(model):
    df_features_importance = pd.DataFrame({
        'Features': model.feature_names_in_,
        'Features importance (in %)': model.feature_importances_ * 100
    })
    return df_features_importance.sort_values(by='Features importance (in %)', ascending=False)

def visualize_SHAP_explanation(model, x_df, file_saving):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_df)
    shap.summary_plot(shap_values, x_df, show=False)
    plt.savefig(file_saving, format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()


#******************************************In Features_impact_analysis.py*********************************************
'''
class FeaturesImpactAnalysis:
    # ...

    def Features_importance(self):
        df_features_importance = compute_features_importance(self.model)
        return df_features_importance

    def SHAP_explainer(self, file_saving):
        visualize_SHAP_explanation(self.model, self.x_df, file_saving)

'''