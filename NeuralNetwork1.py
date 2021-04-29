# A lot of the code is from this tutorial https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=-lcnwG0VXF5h
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loads source dataset into data frame
# data frames allows specific columns to be referenced
dftrain = pd.read_csv('aug_train.csv')
# head() shows first 5 entries
print(dftrain.head())

# Storing the output that will be evaluated against later
output_train  = dftrain.pop('target')
# show dataset without target column
print(dftrain.head())
# show target column, target value is whether or not they are indeed looking for a job
print(output_train)

# Create histogram showing the distribution of training hours across the dataset
#dftrain.gender.hist(bins=10)
#plt.show()

# Creates hostogram showing the males, females and others that need a job
#pd.concat([dftrain, output_train], axis=1).groupby('gender').target.mean().plot(kind='barh').set_xlabel('% need job')
#plt.show()
# seems to show that more females are looking for a job than males

CATEGORICAL_COLUMNS = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
NUMERIC_COLUMNS = ['enrollee_id','city_development', 'training_hours']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # gets list of unique entries for each field
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)