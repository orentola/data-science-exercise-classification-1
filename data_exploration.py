import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

workfolder = "C:\\Users\\orent\\Documents\\Data Science Exercise\\"
all_data_path = workfolder + "hw_analysis.csv"
basic_stats_df = workfolder + "basic_stats_df.csv"

data_all_df = pd.read_csv(all_data_path)

#############################
# Data understanding, analysis, and visualization

basic_stats_df = data_all_df.describe()

data_all_df.hist(bins=50)
plt.show()

# Remove outliers and missing values: remove the X1 = -17.3 and X5 = ""
data_all_df = data_all_df[data_all_df.X1 != -17.3] 
data_all_df = data_all_df.dropna()

# Re-plot the histograms 
data_all_df.hist(bins=50)
plt.show()

# Analyze categorical variables
data_all_df["X7"].value_counts()
data_all_df["X8"].value_counts()
data_all_df["X10"].value_counts()

# See if X10 == B corresponds with the class 1
data_all_df[data_all_df.X10 == 'B'].sum()['Outcome']

# Drop X10 as it corresponds to the positive class
data_all_df = data_all_df.drop(['X10'], axis=1)

correlation_matrix = data_all_df.corr()

# Plot scatter matrix by group, values: X7, X8 and Outcome
sns.set(style="ticks")
sns.pairplot(data_all_df, hue="X7", plot_kws={'alpha':0.05})
plt.show()
