from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder, OrdinalEncoder

def create_features(x_df):
    # features creation code here...
    return x_df

def select_features(x_df, y_df, target, train):
    # feature selection code here...
    return x_df

def encode_scale_features(x_df, categorical_var_OHE, categorical_var_OrdinalEncoding,
                          categorical_var_TE, target, continious_var, encoding_type_cont, train):
    # feature encoding and scaling code here...
    return df_pre_processed



#*******************************************In features_engineering.py*******************************************
'''
class FeaturesEngineering:
    # ...

    def features_creation(self, x_df):
        x_df = create_features(x_df)
        # Rest of your features creation code...

    def filter_selection(self, x_df, y_df, target, train):
        x_df = select_features(x_df, y_df, target, train)
        # Rest of your feature selection code...

    def encoding_scaling(self, x_df, categorical_var_OHE, categorical_var_OrdinalEncoding,
                       categorical_var_TE, target, continious_var, encoding_type_cont, train):
        x_df = encode_scale_features(x_df, categorical_var_OHE, categorical_var_OrdinalEncoding,
                                     categorical_var_TE, target, continious_var, encoding_type_cont, train)
        # Rest of your feature encoding and scaling code...

'''