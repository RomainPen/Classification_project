from unidecode import unidecode
import numpy as np

#TEMPORARY

def lowercase_strings(s):
    return s.lower() if isinstance(s, str) else s

def remove_negative_values(x):
    return np.abs(x) if x < 0 else x

def preprocess_age(age):
    return np.nan if np.abs(age) < 18 else np.abs(age)

def preprocess_gender(gender):
    gender = gender.split("'")[1]
    return np.nan if gender in ["not ent", "unknown"] else gender

def preprocess_age_first_contract(age, tenure_days):
    age_first_contract = age - (tenure_days / 365)
    return np.nan if age_first_contract < 18 else age_first_contract

def preprocess_no_of_recharges(recharges, failed_recharges):
    return np.nan if (recharges - failed_recharges) < 0 else recharges

def preprocess_duration(duration):
    return remove_negative_values(duration)

def preprocess_marque(marque):
    return marque.lower() if isinstance(marque, str) else marque


#********************************************IN Data_cleaning.py file************************************************
'''
class Data_Cleaning:
    # ...

    def treat_anormal_variables(self, df):
        df["CUSTOMER_AGE"] = df["CUSTOMER_AGE"].apply(preprocess_age)
        df["CUSTOMER_GENDER"] = df["CUSTOMER_GENDER"].apply(preprocess_gender)
        df["age_first_contract"] = df.apply(lambda row: preprocess_age_first_contract(row["CUSTOMER_AGE"], row["CONTRACT_TENURE_DAYS"]), axis=1)
        df["NO_OF_RECHARGES_6M"] = df.apply(lambda row: preprocess_no_of_recharges(row["NO_OF_RECHARGES_6M"], row["FAILED_RECHARGE_6M"]), axis=1)
        df[col] = df[col].apply(preprocess_duration)
        return df

    def features_transformation(self, df, external_df):
        df['marque'] = df['CURR_HANDSET_MODE'].map(phones_data.set_index('model')['marque_tel'])
        df = df.drop(["CURR_HANDSET_MODE"], axis=1)
        return df
'''