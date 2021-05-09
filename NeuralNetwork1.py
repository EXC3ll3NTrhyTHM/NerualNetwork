import os
# Reduces python logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Loads source dataset into data frame
# data frames allows specific columns to be referenced
df = pd.read_csv('aug_train.csv')
# Shuffle the data
# sample essentially randomizes data from dataset into new order using frac=1 which treats the whole dataset as the sameple size
df.sample(frac=1)

# Fill in empty values as the string unknown
df['gender'].fillna(value='unknown', inplace=True)
df['enrolled_university'].fillna(value='unknown', inplace=True)
df['education_level'].fillna(value='unknown', inplace=True)
df['major_discipline'].fillna(value='unknown', inplace=True)
df['experience'].fillna(value='unknown', inplace=True)
df['company_size'].fillna(value='unknown', inplace=True)
df['company_type'].fillna(value='unknown', inplace=True)
df['last_new_job'].fillna(value='unknown', inplace=True)

# A lot of this code came from here https://pbpython.com/categorical-encoding.html
df = df.replace({'relevent_experience': {'Has relevent experience':1,'No relevent experience':0}})
df = df.replace({'experience': {'>20':21,'<1':0, 'unknown':0}})
df['experience'] = pd.to_numeric(df['experience'])
df = df.replace({'last_new_job': {'>4':5,'never':0, 'unknown':0}})
df['last_new_job'] = pd.to_numeric(df['last_new_job'])
df['city'] = df['city'].str.replace('city_', '').astype(int)

#print(df['city'].value_counts())

#print(df.dtypes)

gender_cat = pd.get_dummies(df['gender'], prefix='gender')
univer_cat = pd.get_dummies(df['enrolled_university'], prefix='univer')
edu_cat = pd.get_dummies(df['education_level'], prefix='edu')
major_cat = pd.get_dummies(df['major_discipline'], prefix='major')
comp_size_cat = pd.get_dummies(df['company_size'], prefix='comp_size')
comp_type_cat = pd.get_dummies(df['company_type'], prefix='comp_type')

#print(comp_type_cat.head())

df.drop(['gender','enrolled_university','education_level','major_discipline','company_size','company_type'], axis=1,inplace=True)
df = pd.concat((df, gender_cat, univer_cat, edu_cat, major_cat, comp_size_cat, comp_type_cat), axis=1)

train, test = train_test_split(df, test_size=0.2)
#print(test.head())

# Storing the output that will be evaluated against later
output_train  = train.pop('target')
output_test = test.pop('target')

print(train.shape)