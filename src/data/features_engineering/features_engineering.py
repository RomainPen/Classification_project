

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
    
    