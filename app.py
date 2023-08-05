import streamlit as st 
import yaml
import os
import joblib
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from src.data.data_cleaning.data_cleaning import Data_Cleaning
from src.data.features_engineering.features_engineering import FeaturesEngineering
from src.models.model_building.model_building import ModelBuilding
from src.models.model_evaluation.model_evaluation import ModelEvaluation
from src.visualization.model_result_analysis  import ModelResultAnalysis
from src.visualization.Features_impact_analysis import FeaturesImpactAnalysis



# Load settings.yaml :
settings_folder = os.path.join(os.path.dirname(__file__), '', 'config')
settings_path = os.path.join(settings_folder, 'settings.yaml')
with open(settings_path, "r") as settings_file:
        settings = yaml.safe_load(settings_file)
        
# Load the model :
model_folder = os.path.join(os.path.dirname(__file__), '', 'models')
model_path = os.path.join(model_folder, 'scoring_model.pkl')
with open(model_path, 'rb') as model_file:
    scoring_model = joblib.load(model_file) # or pickle.load(file)
 
# Load the image : 
image_folder = os.path.join(os.path.dirname(__file__), '..', 'reports/figures')
image_path = os.path.join(image_folder, 'Customer_Churn.png')
image = Image.open(image_path)    

#download data as csv :
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# Interface building :
# Title :
st.title('Scoring modeling :')
# Image :
st.image(image, caption='Customer churn schema')


# We assume that df was already passed the first 4 step treatment :  
# basic treatment, anormale, feature transformation, handle_missing_value
def main():
    st.header("Dataset clusturing and theme analysis")
    # Import corpus CSV :
    uploaded_file = st.file_uploader("Choose a file (only accept .csv)")
    if uploaded_file is not None :
        df_test = pd.read_csv(uploaded_file)
        # Give column of articles name :
        article_col = st.text_input('Articles column name', 'Article')
        try :
            df_test[["topic", "topic_proba"]] = df_test[article_col].apply(lambda corpus : cluster_similar_documents(corpus))
        except KeyError :
            st.write("This column doesn't exist")
        
        # Analyse the result of the clustering
        topic_count = pd.DataFrame(df_test['topic'].value_counts()).rename(columns={"topic": "nb_of_article"})
        st.dataframe(topic_count)

        #download dat as csv :
        csv = convert_df(topic_count)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='topic_analysis.csv',
            mime='text/csv')
    
        # histogram of output :
        fig, ax = plt.subplots(figsize=(15, 8))
        n_bars = len(topic_count)
        colors = sns.color_palette('husl', n_bars)
        # Shuffle the color list randomly for added randomness
        random.shuffle(colors)
        # Plot the histogram using Matplotlib
        plt.bar(topic_count.index, topic_count['count'], color=colors, width=0.7)
        plt.xlabel('Topics', fontsize=20)
        plt.ylabel('Number of articles', fontsize=20)
        plt.title('Number of articles for each topic', fontsize=20)
        plt.xticks(rotation=50, ha='right') 
        plt.show()
        st.pyplot(fig)






# __name__ :
if __name__ == '__main__' :
    main() 