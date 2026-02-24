# %%
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# %%
# Loading Data 
# Merging Data
# sklearn four steps (review documentation)
# lambda functions

# %%
# load the data, column titles in salary in first row
salary_data = pd.read_csv("2025_salaries.csv", header=1, encoding="latin-1")
stats = pd.read_csv("nba_2025.txt", sep=",", encoding="latin-1")

# %%
# merge the data
merged_data = pd.merge(salary_data, stats, on="Player")
merged_data.head()

# %%
# duplicates in the "Player" column
duplicates = merged_data[merged_data.duplicated(subset="Player", keep=False)]
print(duplicates)

# of the duplicates, you want to keep the stats that correspond to the most games played

# %%
# sklearn four steps
# 1. Create an instance of the model. Ex: mymodel = KMeans(n_clusters=3)
# 2. Fit the model to the data example. Ex: mymodel.fit(X)
# 3. Make predictions using the model. Ex: predictions = mymodel.predict(X)
# 4. Evaluate the model's performance. Ex: score = mymodel.score(X)
# X represents the training dataset

# %%
# General tips:
# leave salaries out, only use it as a color indicator for heatmap
# shape is cluster, color is salary
# lower salaries on left, higher on right
# need to choose variables for x and y axis
# features inside dataset that best display clusters on scatter plot
# 3 clusters across 2 variable types
# how do you pick players that would be good to acquire?
# good players to acquire wouldn't be at the high/low extremes (too expensive vs poor performance)
# they would generally be high performance, low salaries
# good indicators are minutes/games played, average points scored, total rebounds
# these work well because they have a high distribution with large variance, separating good & bad

# %%
# lambda functions
merged_data["Salary_in_thousands"] = merged_data["Salary"].apply(lambda x: x / 1000)
merged_data["High_Salary"] = merged_data["Salary"].apply(lambda x: True if x > 1000000 else False)
