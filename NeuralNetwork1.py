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
dftrain['gender'].fillna(value='unknown', inplace=True)
dftrain['enrolled_university'].fillna(value='unknown', inplace=True)
dftrain['education_level'].fillna(value='unknown', inplace=True)
dftrain['major_discipline'].fillna(value='unknown', inplace=True)
dftrain['experience'].fillna(value='unknown', inplace=True)
dftrain['company_size'].fillna(value='unknown', inplace=True)
dftrain['company_type'].fillna(value='unknown', inplace=True)
dftrain['last_new_job'].fillna(value='unknown', inplace=True)

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

# Column with non-numerical values. will need to associate numerical values to them
CATEGORICAL_COLUMNS = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
NUMERIC_COLUMNS = ['enrollee_id','city_development_index', 'training_hours']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # gets list of unique entries for each field
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, output_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
# This eval input is the exact same as the training input which can lead to the neural network to just memorize the data set and if I were to give it a slightly different dataset to evaluate it might perform a lot worse
eval_input_fn = make_input_fn(dftrain, output_train, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model