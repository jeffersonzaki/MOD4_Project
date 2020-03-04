import pandas as pd
import numpy as np
import statsmodels.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------DATA--------------------------------

freedom_df = pd.read_csv("Data/hfi_cc_2019.csv")[['year', 'hf_score', 'pf_rol', 
                                                  'pf_ss', 'pf_movement','pf_religion', 
                                                  'pf_association', 'pf_expression', 
                                                  'pf_identity', 'ef_government', 'ef_legal', 
                                                  'ef_money', 'ef_trade', 'ef_trade_regulatory']]

countries_regions_df = pd.read_csv("Data/hfi_cc_2019.csv")[["year", "ISO_code", "countries", "region"]]
# ----------------------------------------------------------------


# -----------------------FUNCTIONS------------------------

# Selecting desired year
def year(df, year):
    """
    Able to select rows with
    desired year
    """
    return df.loc[df["year"] == year]


# Dropping features
def drop_feature(df, features):
    return df.drop(columns=features)


# Replacing missing data with a numpy NaN
def replace_missing_to_na(df):
    for strings in df.values:
        df.replace("-", np.NaN, regex=True, inplace=True)


# Function that Normalizes data
def normalize(column):
    upper = column.max()
    lower = column.min()
    y = (column - lower)/(upper-lower)
    return y


# Transforms to sigmoid 
def sigmoid(x):
    e = np.exp(1)
    y = 1/(1+e**(-x))
    return y
# -----------------------------------------------------------


# ------------------DEALING WITH MISSING VALUES----------------

# Finding rows with missing values
missing_values = freedom_df.loc[freedom_df.pf_association == "-"]  # All nulls are located in pf_association column
# Dropping missing values
freedom_df = freedom_df.drop(index=missing_values.index)

# Replacing the human error missing values with numpy NaN
replace_missing_to_na(freedom_df)
freedom_df.dropna(inplace=True)

# --------------------------------------------------------------

# Changing all objects to floats
freedom_df = freedom_df.astype(float)

# Creating two separate dataframes based on year
freedom_2017 = year(freedom_df, 2017)
freedom_2010 = year(freedom_df, 2010)
