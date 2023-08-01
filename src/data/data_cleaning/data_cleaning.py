from unidecode import unidecode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype, is_object_dtype



class Data_Cleaning :
    def __init__(self) :
        pass
        
    
    def basic_treatment(self, df : pd.DataFrame, useless_columns : list) :
        #drop duplicate :
        df = df.drop_duplicates(keep="first")
        #drop useless col:
        df = df.drop(useless_columns, axis=1) #["CONTRACT_KEY"]
        #Handling data formatting: Data formatting involves making sure that the data is in a consistent format. It can be handled by converting data types, changing date formats, etc.
        #lowercase caracter :
        df = df.applymap(lambda s:s.lower() if type(s) == str else s) 
        #drop white space :
        #df = df.applymap(lambda s:s.strip() if type(s) == str else s) 
        #drop multiple(double, triple) space :
        #df = df.applymap(lambda s:s.replace("  ", " ") if type(s) == str else s) 
        #replace " " by "_" :
        #df = df.applymap(lambda s:s.replace(" ", "_") if type(s) == str else s) 
        #remove accent :
        #df = df.applymap(lambda s: unidecode(s) if type(s) == str else s) 
        
        return df
    
    
    def treat_anormal_variables(self, df):
        #age :
        df["CUSTOMER_AGE"] = df["CUSTOMER_AGE"].apply(lambda x : np.nan if np.abs(x)<18 else np.abs(x))
        #customer gender :
        df["CUSTOMER_GENDER"] = df["CUSTOMER_GENDER"].apply(lambda x : x.split("'")[1])
        df["CUSTOMER_GENDER"] = df["CUSTOMER_GENDER"].apply(lambda x: np.nan if (x=="not ent" or x=="unknown") else x)
        #age first contract :
        df["age_first_contract"] = df["CUSTOMER_AGE"] - (df["CONTRACT_TENURE_DAYS"]/365)
        df.loc[df['age_first_contract'] < 18, 'CONTRACT_TENURE_DAYS'] = np.nan
        df = df.drop(["age_first_contract"], axis=1)
        
        # 'NO_OF_RECHARGES_6M' :
        df.loc[(df['NO_OF_RECHARGES_6M']-df['FAILED_RECHARGE_6M']) < 0, 'NO_OF_RECHARGES_6M'] = np.nan
        # proportion de valeurs négatives pour les trois variables concernées
        col = ["INC_OUT_PROP_DUR_MIN_M1", "INC_OUT_PROP_DUR_MIN_M2", "INC_OUT_PROP_DUR_MIN_M3"]
        for elem in col :
            df[elem] = np.where(df[elem] < 0, np.abs(df[elem]), df[elem])
        return df
    
    
    def features_transformation(self, df, external_df) :
        #phones_data = pd.read_excel("marque_telephone.xlsx")
        phones_data = external_df 
        phones_data = phones_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        #duplicates = phones_data.duplicated(subset='model', keep=False) (to drop, not sure)

        # Supprimer les doublons de l'index dans phones_data
        phones_data = phones_data.drop_duplicates(subset='model')

        # Créer une nouvelle colonne 'Marque' dans df
        df['marque'] = df['CURR_HANDSET_MODE'].map(phones_data.set_index('model')['marque_tel'])

        #drop "CURR_HANDSET_MODE":
        df = df.drop(["CURR_HANDSET_MODE"], axis=1)
        return df
        
    
    #split data before, and apply the this function : to review
    def impute_missing_values(self, df : pd.DataFrame, target : str, train :bool) :
        
        if train :
            self.df_train = df.copy()
            #Treat df_train :
            #1.1/ Drop rows and col which are more 60-70% of NaN.
            #If there is 50%-60% or more NaN, drop the column.
            self.df_train_na_col = (pd.DataFrame(((self.df_train.isna().sum(axis=0))/len(self.df_train))*100)).rename(columns={0: 'Sum_NaN_%'})
            #drop col :
            self.df_train = self.df_train.drop(columns=list((self.df_train_na_col[self.df_train_na_col["Sum_NaN_%"]>50]).index))
            #Count the number of NaN in rows :
            #If there is 60-70% or more NaN, drop the row.
            df_train_na_row = (pd.DataFrame(((self.df_train.isna().sum(axis=1))/len(self.df_train.columns))*100)).rename(columns={0: 'Sum_NaN_%'})
            #drop row :
            self.df_train = self.df_train.drop(index=list((df_train_na_row[df_train_na_row["Sum_NaN_%"]>60]).index))
            #reset_index :
            self.df_train = self.df_train.reset_index(drop = True)
            
            #1.2/ dectect col which have NaN
            df_train_na = (pd.DataFrame(self.df_train.isna().sum())).rename(columns={0: 'Sum_NaN'})
            df_train_na_egal_0 = df_train_na[df_train_na['Sum_NaN']==0]
            df_train_na_diff_0 = df_train_na[df_train_na['Sum_NaN']!=0]
            
            #*********************************review this part************************************
            #1.3/Build groupby df for fillna : 
            #select best col for groupby fillna :
            corr_matrix = self.df_train[list(df_train_na_egal_0.index)].corr().abs().loc[target].drop([target], axis=0)
            corr_matrix = pd.concat([corr_matrix, self.df_train[list(corr_matrix.index)].nunique()], axis=1).rename(columns={0: 'nunique'})
            self.df_train_grouby_3var = self.df_train.groupby(["PASS_GRACE_IND_M1", "PASS_GRACE_IND_M2", "PASS_GRACE_IND_M3"])
            self.df_train_grouby_2var = self.df_train.groupby(["PASS_GRACE_IND_M1", "PASS_GRACE_IND_M2"])
            self.df_train_grouby_1var = self.df_train.groupby(["PASS_GRACE_IND_M1"])
            #*************************************************************************************
            
            #1.4/fillna on dataset : 
            # numerical col :
            #fillna with particular value :
            #df_train["num_var"] = df_train["num_var"].fillna("numerical_value")
            #df_train[list_cont_particular_imputation] = df_train[list_cont_particular_imputation].fillna(common value, example = 0)
            #fillna with median :
            for i in (df_train_na_diff_0.index) : # or replace (df_train_na_diff_0.index) by list_median_mode_imputation
                if is_numeric_dtype(self.df_train[i]) and i != target :
                    self.df_train[i] = self.df_train_grouby_3var[i].fillna(self.df_train[i].median()) 
                    self.df_train[i] = self.df_train_grouby_2var[i].fillna(self.df_train[i].median())
                    self.df_train[i] = self.df_train_grouby_1var[i].fillna(self.df_train[i].median())
                    self.df_train[i] = self.df_train[i].fillna(self.df_train[i].median())

            # categorical col :
            #fillna with particular value :
            #df_train["var"] = df_train["var"].fillna("string_value")
            #df_train[list_cat_particular_imputation].fillna("cat_value", inplace=True)
            #***************************************to review*******************************************
            # try to find a manipulation with setting.yaml
            self.df_train['marque'] = self.df_train['marque'].fillna("unknown")
            #*******************************************************************************************
            #fillna with mode :
            for i in (df_train_na_diff_0.index) :  # or replace (df_train_na_diff_0.index) by list_median_mode_imputation
                if is_object_dtype(self.df_train[i]) and i != target :
                    self.df_train[i] = self.df_train_grouby_3var[i].fillna(self.df_train[i].mode()[0]) ##df_train.groupby(["PASS_GRACE_IND_M1", "PASS_GRACE_IND_M2", "PASS_AFTERGRACE_IND_M2"])[i].fillna(df_train[i].mode())
                    self.df_train[i] = self.df_train_grouby_2var[i].fillna(self.df_train[i].mode()[0])
                    self.df_train[i] = self.df_train_grouby_1var[i].fillna(self.df_train[i].mode()[0])
                    self.df_train[i] = self.df_train[i].fillna(self.df_train[i].mode()[0])
            
            return self.df_train
          
                
        else :
            self.df_val = df.copy()
            # Treat df_val :
            #1.1/ Drop rows and col which are more 60-70% of NaN.
            #If there is 50%-60% or more NaN, drop the column.
            #drop col :
            self.df_val.drop(columns=list((self.df_train_na_col[self.df_train_na_col["Sum_NaN_%"]>50]).index), inplace=True)
            #Count the number of NaN in rows :
            #If there is 60-70% or more NaN, drop the row.
            #useless for row of df_val 
            
            #***********************************to review****************************************
            #1.1/ verify if columns (which we use for groupby) have NaN or not :
            groupby_columns = ["PASS_GRACE_IND_M1", "PASS_GRACE_IND_M2", "PASS_GRACE_IND_M3"]
            #************************************************************************************
            
            # if there is NaN, fillna them by the median/mean or mode from df_train
            #numerical col :
            for i in groupby_columns :
                self.df_val[i] = self.df_val[i].fillna(self.df_train[i].median())
            #cat col :
            for i in groupby_columns :
                self.df_val[i] = self.df_val[i].fillna(self.df_train[i].mode())
                
            #1.2/dectect col which have NaN
            df_val_na = (pd.DataFrame(self.df_val.isna().sum())).rename(columns={0: 'Sum_NaN'})
            #df_val_na_egal_0 = df_val_na[df_val_na['Sum_NaN']==0]
            df_val_na_diff_0 = df_val_na[df_val_na['Sum_NaN']!=0]
            
            
            #*********************************************to review**************************************************
            #1.2/fillna on dataset :
            #numerical col :
            #fillna with particular value :
            #df_val["var"] = df_val["var"].fillna("numerical_value")
            #df_val[list_cont_particular_imputation] = df_val[list_cont_particular_imputation].fillna(0)
            #fillna with median : 
            def imputate_missing_3val_num(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_3var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"], "PASS_GRACE_IND_M2"==df["PASS_GRACE_IND_M2"], "PASS_GRACE_IND_M3"==df["PASS_GRACE_IND_M3"]))[[elem for elem in self.df_val.columns if is_numeric_dtype(self.df_val[elem])]].agg("median"))[i]
                else :
                    return df[i]

            def imputate_missing_2val_num(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_2var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"], "PASS_GRACE_IND_M2"==df["PASS_GRACE_IND_M2"]))[[elem for elem in self.df_val.columns if is_numeric_dtype(self.df_val[elem])]].agg("median"))[i]
                else :
                    return df[i]

            def imputate_missing_1val_num(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_1var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"]))[[elem for elem in self.df_val.columns if is_numeric_dtype(self.df_val[elem])]].agg("median"))[i]
                else :
                    return df[i]
            
            for i in (df_val_na_diff_0.index) : #or list_median_mode_imputation
                if is_numeric_dtype(self.df_val[i]) and i != target :
                    self.df_val[i] = self.df_val.apply(imputate_missing_3val_num, axis=1)
                    self.df_val[i] = self.df_val.apply(imputate_missing_2val_num, axis=1)
                    self.df_val[i] = self.df_val.apply(imputate_missing_1val_num, axis=1)
                    self.df_val[i] = self.df_val[i].fillna(self.df_train[i].median())
                    
            # caterical col :
            #fillna with particular value :
            #df_val["var"] = df_val["var"].fillna("categorical_value")
            self.df_val['marque'].fillna("unknown", inplace=True)
            #fillna with mode :
            def imputate_missing_3val_cat(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_3var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"], "PASS_GRACE_IND_M2"==df["PASS_GRACE_IND_M2"],
                                                            "PASS_GRACE_IND_M3"==df["PASS_GRACE_IND_M3"]))[[elem for elem in self.df_val.columns if is_object_dtype(self.df_val[elem])]].agg("mode"))[i][0]
                else :
                    return df[i]
                
            def imputate_missing_2val_cat(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_2var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"],
                                                            "PASS_GRACE_IND_M2"==df["PASS_GRACE_IND_M2"]))[[elem for elem in self.df_val.columns if is_object_dtype(self.df_val[elem])]].agg("mode"))[i][0]
                else :
                    return df[i]

            def imputate_missing_1val_cat(df) :
                if pd.isna(df[i]) :
                    return (self.df_train_grouby_1var.get_group(("PASS_GRACE_IND_M1"==df["PASS_GRACE_IND_M1"]))[[elem for elem in self.df_val.columns if is_object_dtype(self.df_val[elem])]].agg("mode"))[i][0]
                else :
                    return df[i]

            for i in (df_val_na_diff_0.index) : #or list_median_mode_imputation
                if is_object_dtype(self.df_val[i]) and i != target :
                    if self.df_val[i].isnull().values.any() :
                        self.df_val[i] = self.df_val.apply(imputate_missing_3val_cat, axis=1)
                        self.df_val[i] = self.df_val.apply(imputate_missing_2val_cat, axis=1)
                        self.df_val[i] = self.df_val.apply(imputate_missing_1val_cat, axis=1)
                        self.df_val[i] = self.df_val[i].fillna(self.df_train[i].mode())
            
            return self.df_val
            #******************************************************************************************************
        
        