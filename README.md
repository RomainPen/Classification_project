# Classification Project 

## Introduction
Welcome to the Classification Project! This machine learning project aims to build a classification model for a telecommunication dataset to predict customer churn. The project is structured to facilitate easy development, understanding, and collaboration. This README document provides an overview of the project structure, its components, and instructions to get started.

## Project Structure
The project is organized into the following directories and files:

1. **config:** This directory contains configuration files for the project.
   - **app_config_sample.ini:** An empty sample configuration file for the application.
   - **hyperparameters.yaml:** YAML file for storing hyperparameters used in the model.
   - **logging_sample.conf:** An empty sample logging configuration file.
   - **settings.yaml:** YAML file for storing general project settings.

2. **data:** This directory contains data used in the project. It is further divided into subdirectories:
   - **external:** Contains data from external sources.
      - **marque_telephone.csv:** CSV file with data about phone brands.
      - **marque_telephone.xlsx:** Excel file with data about phone brands.
   - **interim_demo:** Stores intermediate data used during the development process.
      - **x_test1.csv:** CSV file with interim test data 1.
      - **x_test2.csv:** CSV file with interim test data 2.
   - **processed:** Contains processed and pre-processed data files.
      - **df_train.csv:** CSV file with pre-processed training data.
      - **df_val.csv:** CSV file with pre-processed validation data.
   - **raw:** Stores raw data files.
      - **base_projet_teleco.csv:** CSV file with raw telecommunication data.

3. **docs:** This directory contains project documentation. It includes a subdirectory "Project_structure_examples" with some example images and files.
   - **Project_structure_examples:**
      - **Project_structure1.jpg:** Example image of the project's directory structure.
      - **Project_structure2.png:** Another example image of the project's directory structure.
      - **Unknown_file.txt:** A text file of unknown content.

4. **models:** This directory is used to store trained machine learning models.
   - **scoring_model.pkl:** Pickle file containing a trained scoring model.

5. **notebooks:** This directory contains Jupyter notebooks used for various tasks related to the project.
   - **Creation_variables.ipynb:** Notebook for creating and engineering variables.
   - **Notebook_code.ipynb:** Main notebook containing project code and analysis.
   - **test.ipynb:** Notebook for testing various components.

6. **references:** This directory holds reference materials for the project.
   - **Classification_project_structure:**
      - **Classification_project_structure_OFF.txt:** Offline classification project structure document.
      - **Classification_project_structure_test.txt:** Test classification project structure document.
   - **install_guide.txt:** Guide for installing dependencies and setting up the environment.

7. **reports:** This directory contains various reports and analysis results. It includes subdirectories for different analyses.
   - **EDA:** Contains exploratory data analysis figures.
      - **corr_matrix_df_train.png:** Correlation matrix plot of the training data.
      - **target_distrib_groupby_CUSTOMER_GENDER.png:** Target distribution plot grouped by customer gender.
      - **target_distrib_groupby_marque.png:** Target distribution plot grouped by phone brand.
   - **Features_impact_analysis:** Contains files related to feature impact analysis.
      - **SHAP_summary_plot.pdf:** Summary plot of SHAP (SHapley Additive exPlanations) values.
   - **figures:** Contains additional figures used in the project.
      - **Customer_Churn.png:** Customer churn plot.
      - **Customer_Churn2.png:** Another customer churn plot.
      - **customer_churn3.webp:** WebP image of customer churn (Note: The extension should be ".webm" instead of ".webp").
      - **streamlit-APP-2023-07-19-10-07-63.webm:** WebM video of a Streamlit application.
   - **Model_result_analysis:** Contains model evaluation results.
      - **CONFUSION_MATRIX.png:** Confusion matrix plot.
      - **LEARNING_CURVE.png:** Learning curve plot.
      - **LIFT_CURVE.png:** Lift curve plot.
      - **ROC_AUC_curve.png:** ROC AUC curve plot.
   - **PENICHON_BOUMEZAOUED_GEFFLOT.pptx:** PowerPoint presentation related to the project.
   - **SCORING.pdf:** Scoring document for the project.

8. **src:** This directory contains the source code of the project, organized into subdirectories for different modules.
   - **data:**
      - **data_cleaning:**
         - **data_cleaning.py:** Module for data cleaning functions.
      - **features_engineering:**
         - **features_engineering.py:** Module for feature engineering functions.
   - **models:**
      - **model_building:**
         - **model_building.py:** Module for building machine learning models.
      - **model_evaluation:**
         - **model_evaluation.py:** Module for evaluating machine learning models.
   - **utils:** (Currently empty) This directory might store common utility functions.
      - **common_utils.py:** Common utility functions.
      - **data_cleaning_utils.py:** data_cleaning utility functions.
      - **EDA_utils.py:** EDA utility functions.
      - **feature_engineering_utils.py:** feature_engineering utility functions.
      - **Features_impact_analysis_utils.py:** Features_impact_analysis utility functions.
      - **model_building_utils.py:** model_building utility functions.
      - **model_evaluation_utils.py:** model_evaluation utility functions.
      - **model_result_analysis_utils.py:** model_result_analysis utility functions.
   - **visualization:**
      - **EDA.py:** Module for data visualization and exploratory data analysis functions.
      - **Features_impact_analysis.py:** Module for feature impact analysis functions.
      - **model_result_analysis.py:** Module for model result analysis functions.

9. **tests:** This directory contains test scripts for the project.
      Run the tests using the pytest command :
```bash
      pytest tests/test_file.py
```
      Example of testing data_cleaning.py :
```bash
      pytest tests/test_data_cleaning.py
```
   - **test_data_cleaning.py:** Test script for data cleaning module.
   - **test_EDA.py:** Test script for exploratory data analysis module.
   - **test_Features_impact_analysis.py:** Test script for feature impact analysis module.
   - **test_feature_engineering.py:** Test script for feature engineering module.
   - **test_model_building.py:** Test script for model building module.
   - **test_model_evaluation.py:** Test script for model evaluation module.
   - **test_model_result_analysis.py:** Test script for model result analysis module.

10. **Other files:** Additional project-related files and configuration files.
   - **.gitignore:** File that specifies which files and directories to ignore when using Git version control.
   - **app.py:** Main application file.
   - **main.py:** Main script to run the project.
   - **Pipfile:** Pipenv file specifying project dependencies.
   - **Pipfile.lock:** Pipenv lock file with exact dependency versions.
   - **pyproject.toml:** (Currently empty) File for project-specific configuration.
   - **pytest_sample.ini:** Sample configuration file for pytest.
   - **README.md:** This README document.
   - **requirements.txt:** File specifying project dependencies.
   - **tox.ini:** (Currently empty) Configuration file for tox testing tool.

## Getting Started
To run this project on your local machine, follow these steps:

1. Clone the repository:
```bash
   git clone https://github.com/RomainPen/Classification_project.git
   cd Classification_project
```

2. create the new env : 
 manager, pip:
 ```bash
  py -m pipenv install
```

3. Install the required dependencies using the package
 manager, pip:
```bash
   pipenv shell 
   #or pipenv install -r pipfile.txt #(or requirements.txt)
```

4. Navigate to the project root directory and run the main.py script:
```bash
   python main.py
```

5. Access the Streamlit web app using the provided link: https://nppm6uqcxxf3vmokesgbql.streamlit.app/

## Usage and Contribution
Feel free to use and modify this project according to your needs. If you wish to contribute to the project, follow these steps:

1. Fork the repository on GitHub.

2. Create a new branch with a descriptive name for your changes.

3. Make your changes and commit them with appropriate messages.

4. Push your branch to your forked repository.

5. Create a pull request to merge your changes into the main project repository.

## Acknowledgments
We would like to thank all contributors and acknowledge any third-party data or tools used in this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have any questions or suggestions, feel free to contact the project maintainers:
- PENICHON Romain - romain.pen.pro16@gmail.com

We hope you find this project useful and enjoy exploring the world of machine learning and classification!