import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def plot_categorical_distribution(dataframe, target, var_cat, file_saving):
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
    plt.title("Distribution of '" + var_cat + "' by '" + target + "'", fontsize=22)
    plt.legend(fontsize=14)
    plt.xlabel(target, fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.savefig(file_saving + var_cat + '.png', format='png')
    plt.close()

def plot_numerical_distribution(dataframe, target, var_num, file_saving):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    sns.histplot(dataframe[dataframe[target] == 0][var_num], ax=ax1)
    sns.histplot(dataframe[dataframe[target] == 1][var_num], ax=ax2)
    ax1.set_title("Distribution of " + var_num + f" \n for '{target}' = 0")
    ax2.set_title("Distribution of " + var_num + f" \n for '{target}' = 1")
    plt.savefig(file_saving + var_num + '.png', format='png')
    plt.close()

def plot_correlation_matrix(df, target, file_saving):
    correlation_matrix = df.select_dtypes(include=np.number).corr()

    threshold = 0.15
    target_correlations = correlation_matrix[target][(correlation_matrix[target] > threshold) | (correlation_matrix[target] < -threshold)]

    filtered_df = df[target_correlations.index]
    filtered_correlation_matrix = filtered_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", annot_kws={"ha": 'center'})
    plt.title(f"Correlation Matrix with Variables Correlated to '{target}' (correlation > 0.15)")
    plt.savefig(file_saving, format='png')
    plt.close()



#****************************************************In EDA.py file*************************************************

'''
class EDA:
    # ...

    def target_feature_distribution_groupby_categorical_features(self, file_saving):
        for i in self.var_cat:
            plot_categorical_distribution(self.df, self.target, i, file_saving)

    def target_feature_distribution_groupby_numerical_features(self, file_saving):
        for i in self.var_num[:2]:
            plot_numerical_distribution(self.df, self.target, i, file_saving)

    def correlation_matrix(self, file_saving):
        plot_correlation_matrix(self.df, self.target, file_saving)


'''









