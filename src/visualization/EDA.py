import numpy as np
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_object_dtype

class EDA:
    def __init__(self, df, target:str):
        self.df = df
        self.target = target
        # Sélection des colonnes numériques à l'exclusion de la variable cible
        self.var_num = df.select_dtypes(include=np.number).columns.tolist()
        self.var_num.remove(target)

        # Sélection des colonnes catégorielles à l'exclusion de la variable CURR_HANDSET_MODE
        self.var_cat = df.select_dtypes(include=object).columns.tolist()
        
    
    def general_summary_skimpy(self):
        #skim(self.df)
        return None
    
    
    def distrib_for_cat_by_target(self, var_cat: list, dataframe, target: str, file_saving):
        temp = dataframe.copy()
        temp['Frequency'] = 0
        counts = temp.groupby([target, var_cat]).count()
        freq_per_group = counts.div(counts.groupby(target).transform('sum')).reset_index()
        g = sns.catplot(x=target, y="Frequency", hue=var_cat, data=freq_per_group, kind="bar",
                    height=8, aspect=2, legend=False)
        ax = g.ax
        for p in ax.patches:
            ax.annotate(f"{p.get_height()*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=14, color='black', xytext=(0, 20),
                        textcoords='offset points')
        plt.title("Distribution de '" + var_cat + "' par 'Cible'", fontsize=22)
        plt.legend(fontsize=14)
        plt.xlabel(target, fontsize=18)
        plt.ylabel('Fréquence', fontsize=18)
        plt.savefig(file_saving + var_cat + '.png', format='png')
        plt.close()
    
    def target_feature_distribution_groupby_categorical_features(self, file_saving):
        for i in self.var_cat:
            self.distrib_for_cat_by_target(i, self.df, self.target, file_saving)
            
    
    
    def distrib_for_num_by_target(self, var_num: list, dataframe, target: str, file_saving):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
        sns.histplot(dataframe[dataframe[target] == 0][var_num], ax=ax1)
        sns.histplot(dataframe[dataframe[target] == 1][var_num], ax=ax2)
        ax1.set_title("Distribution de la variable " + var_num + f" \n pour '{target}' = 0")
        ax2.set_title("Distribution de la variable " + var_num + f" \n pour '{target}' = 1")
        plt.savefig(file_saving + var_num + '.png', format='png')
        plt.close()
    
    def target_feature_distribution_groupby_numerical_features(self, file_saving):
        for i in self.var_num[:2] :
            self.distrib_for_num_by_target(i, self.df, self.target, file_saving)
    
    
    
    def correlation_matrix(self, file_saving):
        # Matrice de corrélation
        correlation_matrix = self.df[[elem for elem in self.df.columns if is_numeric_dtype(self.df[elem])]].corr()

        # Sélection des variables les plus corrélées à AFTERGRACE_FLAG avec une corrélation supérieure à 0.15
        threshold = 0.15
        target_correlations = correlation_matrix[self.target][(correlation_matrix[self.target] > threshold) | (correlation_matrix[self.target] < -threshold)]

        # Filtrage du DataFrame original pour les variables sélectionnées
        filtered_df = self.df[target_correlations.index]

        # Création d'une nouvelle matrice de corrélation avec les variables sélectionnées
        filtered_correlation_matrix = filtered_df.corr()

        # Création de la heatmap de la matrice de corrélation filtrée
        plt.figure(figsize=(10, 8))
        sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", annot_kws={"ha": 'center'})
        plt.title(f"Matrice de corrélation avec variables corrélées à {self.target} (corrélation > 0.15)")
        plt.savefig(file_saving, format='png')
        plt.close()
        