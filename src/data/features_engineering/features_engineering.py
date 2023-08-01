import pandas as pd 
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer  
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder, OrdinalEncoder



class FeaturesEngineering :
    def __init__(self):
        pass
    
    def features_creation(self, x_df):
        # création de 4 variables (comme on cherche les churners dans les 2 mois)
        x_df["FLAG_RECHARGE_M1"] = x_df["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 0 <= x <= 31 else 0)
        x_df["FLAG_RECHARGE_M2"] = x_df["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 32 <= x <= 62 else 0)
        x_df["FLAG_RECHARGE_M3"] = x_df["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if 63 <= x <= 92 else 0)
        x_df["FLAG_RECHARGE_PLUS_M3"] = x_df["RECENCY_OF_LAST_RECHARGE"].apply(lambda x : 1 if x >= 93 else 0) #plus loin que M3
        
        # approche via les balances : si balance M2 > M1 et balance M3 > M2 alors il a eu plusieurs recharges sur les 3 mois
        # marche que si balance = reste des recharges
        for index, row in x_df.iterrows():
            if row["BALANCE_M2"] > row["BALANCE_M1"] and row["BALANCE_M3"] > row["BALANCE_M2"]:
                x_df.at[index, "AVERAGE_MULTIPLE_RECHARGE_M1_M2_M3"] = 1
            else :
                x_df.at[index, "AVERAGE_MULTIPLE_RECHARGE_M1_M2_M3"] = 0
                
        # si quelque chose entrer 1 sinon 0
        for index, row in x_df.iterrows() :
            if row["INC_DURATION_MINS_M1"] + row["INC_PROP_SMS_CALLS_M1"] == 0 :
                x_df.at[index, "FLAG_IN_M1"] = 0
            else :
                x_df.at[index, "FLAG_IN_M1"] = 1
            
            if row["INC_DURATION_MINS_M2"] + row["INC_PROP_SMS_CALLS_M2"] == 0 :
                x_df.at[index, "FLAG_IN_M2"] = 0
            else :
                x_df.at[index, "FLAG_IN_M2"] = 1
            
            if row["INC_DURATION_MINS_M3"] + row["INC_PROP_SMS_CALLS_M3"] == 0 :
                x_df.at[index, "FLAG_IN_M3"] = 0
            else :
                x_df.at[index, "FLAG_IN_M3"] = 1
                
        # si quelque chose sort 1 sinon 0
        for index, row in x_df.iterrows() :
            if row["OUT_DURATION_MINS_M1"] + row["OUT_SMS_NO_M1"] + row["OUT_INT_DURATION_MINS_M1"] + row["OUT_888_DURATION_MINS_M1"] + row["OUT_VMACC_NO_CALLS_M1"] == 0:
                x_df.at[index, "FLAG_OUT_M1"] = 0
            else :
                x_df.at[index, "FLAG_OUT_M1"] = 1

            if row["OUT_DURATION_MINS_M2"] + row["OUT_SMS_NO_M2"] + row["OUT_INT_DURATION_MINS_M2"] == 0 + row["OUT_888_DURATION_MINS_M2"] + row["OUT_VMACC_NO_CALLS_M2"] == 0 :
                x_df.at[index, "FLAG_OUT_M2"] = 0
            else :
                x_df.at[index, "FLAG_OUT_M2"] = 1

            if row["OUT_DURATION_MINS_M3"] + row["OUT_SMS_NO_M3"] + row["OUT_INT_DURATION_MINS_M3"] + row["OUT_888_DURATION_MINS_M3"] + row["OUT_VMACC_NO_CALLS_M3"] == 0 :
                x_df.at[index, "FLAG_OUT_M3"] = 0
            else :
                x_df.at[index, "FLAG_OUT_M3"] = 1
                
        # type de contrat : ancien ou nouveau (règle : si supérieur à 2 ans vieux sinon nouveau)
        for index, row in x_df.iterrows() :
            if row["CONTRACT_TENURE_DAYS"] > 730 :
                x_df.at[index, "OLD_CONTRACT"] = 1
            else :
                x_df.at[index,"OLD_CONTRACT"] = 0
                
        return x_df
    
    
    def filter_selection(self, x_df, y_df, target, train : bool):
        # Variables catégortielles test du Chi 2:
        
        if train : 
            df_train = pd.concat([x_df, y_df], axis=1, join="inner")
            info_types = pd.DataFrame(df_train.dtypes)
            list_var_cat = info_types[info_types[0]=="object"].index.tolist()
            self.list_col_to_drop = []
            for v in list_var_cat:
                if v!=target:
                    # Création de la table de contingence
                    cont = df_train[[v, target]].pivot_table(index=v, columns=target, aggfunc=len).fillna(0).copy().astype(int)
                    st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(cont)
                    #col to drop :
                    if st_p >= 0.05 :
                        self.list_col_to_drop.append(v)
            
            #Variable num, test de Student :
            info_df_num = df_train.describe()
            for v in info_df_num.columns.tolist():
                if v!= target:
                    a=list(df_train[df_train[target]==0][v])
                    b=list(df_train[df_train[target]==1][v])
                    st_test, st_p = st.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='omit')
                    #col to drop :
                    if st_p >= 0.05 :
                        self.list_col_to_drop.append(v)     
            x_df = x_df.drop(self.list_col_to_drop, axis=1)
        else :
            x_df = x_df.drop(self.list_col_to_drop, axis=1)
        return x_df
    
    
    def encoding_scaling(self, x_df :pd.DataFrame, categorical_var_OHE:list,
                       categorical_var_OrdinalEncoding:dict, categorical_var_TE: list, target, continious_var:list,
                       encoding_type_cont, train : bool) -> pd.DataFrame :
        """
        Summary: This method aim to encode and scale a dataframe (df)

        Args:
            x_df (pd.DataFrame): cleaned dataframe
            categorical_var_OHE (list): list of categorical columns who will be encoded with OHE
                                        Ex : ["col1", "col2"]
            categorical_var_OrdinalEncoding (dict): dict of categorical columns who will be encoded with ordinal encoding.
                                                    Ex : {'col1': {'a':0, 'b':1, 'c':2},
                                                          'col2': {'d': 2, 'e':1, 'f':0}}
            categorical_var_TE (list): list of categorical columns who will be encoded with Target encoder
                                        Ex : ["col1", "col2"]
            target (series): y_train for training TE
            continious_var (list): list of continious columns who will be scaled with "encoding_type_cont"
                                    Ex : ["col1", "col2"]
            encoding_type_cont (_type_): types of scaling. MinMaxScaler() or StandardScaler()
            train (bool): Ask if you are encoding/scaling a traning set or not (if yes we fit_transform it, else we transform it)

        Returns:
            pd.DataFrame: encoded/scaled dataframe
        """
        df_pre_processed = x_df.copy()
        
        if train == True :
            #continious var encoding :
            if len(continious_var) != 0 :
                #continious var :
                self.scaler = encoding_type_cont #StandardScaler() #or MinMaxScaler()
                df_pre_processed[continious_var] = self.scaler.fit_transform(df_pre_processed[continious_var])    
           
            #categorical encoding :
            if len(categorical_var_OHE) != 0 :
            #categorical var : OHE
                self.enc_OHE = OneHotEncoder(drop='first', sparse=False).fit(df_pre_processed[categorical_var_OHE])
                encoded = self.enc_OHE.transform(df_pre_processed[categorical_var_OHE])
                encoded_df = pd.DataFrame(encoded,columns=self.enc_OHE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_OHE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df], axis=1)
               
            if len(categorical_var_OrdinalEncoding) != 0 :
            #categorical var : Ordinal input example -> {"var" : {'c':0,'b':1,'a':2}}
                for i in range(len(categorical_var_OrdinalEncoding)) :
                    var = list(categorical_var_OrdinalEncoding.keys())[i]
                    self.enc_ordinal = OrdinalEncoder(cols=[var], return_df=True, mapping=[{'col':var,'mapping':categorical_var_OrdinalEncoding[var]}])
                    df_pre_processed[var] = self.enc_ordinal.fit_transform(df_pre_processed[var])
            
            if len(categorical_var_TE) != 0 :
            #categorical var : Target encoding
                self.enc_TE = TargetEncoder().fit(df_pre_processed[categorical_var_TE], target)
                encoded_TE = self.enc_TE.transform(df_pre_processed[categorical_var_TE])
                encoded_df_TE = pd.DataFrame(encoded_TE,columns=self.enc_TE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_TE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df_TE], axis=1)
            
        else :
            #continious encoding :
            if len(continious_var) != 0 :
                #continious var :
                df_pre_processed[continious_var] = self.scaler.transform(df_pre_processed[continious_var])
             
            #categorical encoding :  
            if len(categorical_var_OHE) != 0 :
                #categorical var : OHE
                encoded2 = self.enc_OHE.transform(df_pre_processed[categorical_var_OHE])
                encoded_df2 = pd.DataFrame(encoded2,columns=self.enc_OHE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_OHE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df2], axis=1)
           
            if len(categorical_var_OrdinalEncoding) != 0 :
            #categorical var : Ordinal input example -> {"var" : {'c':0,'b':1,'a':2}}
                for i in range(len(categorical_var_OrdinalEncoding)) :
                    var = list(categorical_var_OrdinalEncoding.keys())[i]    
                    df_pre_processed[var] = self.enc_ordinal.transform(df_pre_processed[var])
             
            if len(categorical_var_TE) != 0 :
                #categorical var : Target Encoding
                encoded2_TE = self.enc_TE.transform(df_pre_processed[categorical_var_TE])
                encoded_df2_TE = pd.DataFrame(encoded2_TE,columns=self.enc_TE.get_feature_names_out())
                df_pre_processed.drop(categorical_var_TE, axis=1, inplace=True)
                df_pre_processed = pd.concat([df_pre_processed, encoded_df2_TE], axis=1)
            
        return df_pre_processed
        