
"""
Created on Sun Aug 13 12:40:00 2023

@author: nagashree k d
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
#from scipy import stat
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot

dir(pd)

# Read data into Python
steel = pd.read_csv(r"C:\Users\nagashree k d\Documents\product tmt.csv")

from sqlalchemy import create_engine
# Credentials to connect to Database
user = 'root'  # user name
pw = 'N%40gashree'  # password
db = 'tmt'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# to_sql() - function to push the dataframe onto a SQL table.
steel.to_sql('steelrod_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from steelrod_tbl;'
df= pd.read_sql_query(sql, engine)

data_types = steel.dtypes


steel.info
steel.isna
# Read data into Python
steel.shape

df = pd.DataFrame(steel)

# Check for duplicates
duplicates = df[df.duplicated()]
print(duplicates)

df.drop_duplicates()


# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
df.Quantity.mean()
df.Quantity.median()
df.Quantity.mode()


df.Rate.mean() # '.' is used to refer to the variables within object
df.Rate.median()
df.Rate.mode()

df.Value.mean()            # '.' is used to refer to the variables within object
df.Value.median()
df.Value.mode()


# Measures of Dispersion / Second moment business decision
df.Quantity.var() # variance
df.Quantity.std() # standard deviation
range = max( df.Quantity) - min(df.Quantity) # range
range

# Measures of Dispersion / Second moment business decision
df.Rate.var() # variance
df.Rate.std() # standard deviation
range = max(df.Rate) - min(df.Rate) # range
range


# Measures of Dispersion / Second moment business decision
df.Value.var() # variance
df.Value.std() # standard deviation
range = max(df.Value) - min(df.Value) # range
range


# Third moment business decision
df.Quantity.skew()
df.Rate.skew()

df.Value.skew()



# Fourth moment business decision
df.Quantity.kurt()
df.Rate.kurt()

df.Value.kurt()

# Histogram
plt.hist(df.Rate) # histogram
plt.hist(df.Rate, bins = [600, 680, 710, 740, 780], color = 'green', edgecolor="red") 

plt.hist(df.Value)
plt.hist(df.Value, color='green', edgecolor = "black", bins = 6)
 
plt.hist(df.Quantity)
plt.hist(df.Quantity, color='red', edgecolor = "black", bins = 6)

import seaborn as sns
# Density Plot
sns.kdeplot(df.Rate) # Density plot
sns.kdeplot(df.Rate, bw = 0.5 , fill = True)


# Density Plot
sns.kdeplot(df.Value) # Density plot
sns.kdeplot(df.Value, bw = 0.5 , fill = True)

# Density Plot
sns.kdeplot(df.Quantity) # Density plot
sns.kdeplot(df.Quantity, bw = 0.5 , fill = True)

#####Data preprocessing#####

## Outliers detection ####
# Let's find outliers in Rate
sns.boxplot(df.Rate) 
#  some outliers in Rate column

# Let's find outliers in Value
sns.boxplot(df.Value)
##  some outliers in Value column

# Let's find outliers in Quantity
sns.boxplot(df.Quantity)
##  some ouliers in Quantity column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Rate'].quantile(0.75) - df['Rate'].quantile(0.25)

lower_limit = df['Rate'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Rate'].quantile(0.75) + (IQR * 1.5)


# Detection of outliers (find limits for salary based on IQR)
IQR = df['Value'].quantile(0.75) - df['Value'].quantile(0.25)

lower_limit = df['Value'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Value'].quantile(0.75) + (IQR * 1.5)


# Detection of outliers (find limits for salary based on IQR)
IQR = df['Quantity'].quantile(0.75) - df['Quantity'].quantile(0.25)

lower_limit = df['Quantity'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Quantity'].quantile(0.75) + (IQR * 1.5)


############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Rate > upper_limit, True, np.where(df.Rate < lower_limit, True, False))
df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Rate)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Rate'] > upper_limit, upper_limit, np.where(df['Rate'] < lower_limit, lower_limit, df['Rate'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Rate'])

df_s = winsor_iqr.fit_transform(df[['Rate']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Rate)



############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Value > upper_limit, True, np.where(df.Value < lower_limit, True, False))
df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Value)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Value'] > upper_limit, upper_limit, np.where(df['Value'] < lower_limit, lower_limit, df['Value'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Value'])

df_s = winsor_iqr.fit_transform(df[['Value']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Value)


############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Quantity > upper_limit, True, np.where(df.Quantity < lower_limit, True, False))
df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Quantity)

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Quantity'] > upper_limit, upper_limit, np.where(df['Quantity'] < lower_limit, lower_limit, df['Quantity'])))
sns.boxplot(df.df_replaced)


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Quantity'])

df_s = winsor_iqr.fit_transform(df[['Quantity']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(df_s.Quantity)

# Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking whether data is normally distributed
stats.probplot(df.Rate, dist = "norm", plot = pylab) ## Data is not completely normally distributeed

stats.probplot(df.Value, dist = "norm", plot = pylab) ## Data is not normally distributed

stats.probplot(df.Quantity, dist = "norm", plot = pylab) ## Data is not normally distributed

# Transformation to make Rate variable normal
stats.probplot(np.log(df.Rate), dist = "norm", plot = pylab)

# Transformation to make Value variable normal
stats.probplot(np.log(df.Value), dist = "norm", plot = pylab)

# Transformation to make Quantity variable normal
stats.probplot(np.log(df.Quantity), dist = "norm", plot = pylab)

# Automated EDA methods

