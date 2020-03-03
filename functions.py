import pandas as pd
import statsmodels.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


freedom_df = pd.read_csv("Data/hfi_cc_2019.csv")[['year', 'hf_score', 'pf_rol', 
                                                  'pf_ss', 'pf_movement','pf_religion', 
                                                  'pf_association', 'pf_expression', 
                                                  'pf_identity', 'ef_government', 'ef_legal', 
                                                  'ef_money', 'ef_trade', 'ef_trade_regulatory']]

countries_regions_df = pd.read_csv("Data/hfi_cc_2019.csv")[["year","ISO_code", "countries", "region"]]

# Selecting desired year
def year(df, year):
    """
    Able to select rows with
    desired year
    """
    return df.loc[df["year"] == year]


freedom_2017 = year(freedom_df, 2017)
freedom_2010 = year(freedom_df, 2010)

# Finding rows with missing values
missing_values = freedom_df.loc[freedom_df.pf_association == "-"]  # All nulls are located in pf_association column
# Dropping missing values
freedom_df = freedom_df.drop(index=missing_values.index)

# Changing all objects to floats
# freedom_df = freedom_df.astype(float)

# Dropping features
def drop_feature(df, features):
    return df.drop(columns=features)
