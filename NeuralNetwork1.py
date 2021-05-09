import os
# Reduces python logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
#df.sample(frac=1)

# Fill in empty values as the string unknown
df['gender'].fillna(value='unknown', inplace=True)
df['enrolled_university'].fillna(value='unknown', inplace=True)
df['education_level'].fillna(value='unknown', inplace=True)
df['major_discipline'].fillna(value='unknown', inplace=True)
df['experience'].fillna(value='unknown', inplace=True)
df['company_size'].fillna(value='unknown', inplace=True)
df['company_type'].fillna(value='unknown', inplace=True)
df['last_new_job'].fillna(value='unknown', inplace=True)

# A lot of categorical data encoding came from here https://pbpython.com/categorical-encoding.html
df = df.replace({'relevent_experience': {'Has relevent experience':1,'No relevent experience':0}})
# Made unknown 0 since there werent that many unknown entries not exactly accurate though, making these indicator colunms would be more accurate but would add a considerable amout
# of extra columns to my dataset without really adding that much value
df = df.replace({'experience': {'>20':21,'<1':0, 'unknown':0}})
df['experience'] = pd.to_numeric(df['experience'])
df = df.replace({'last_new_job': {'>4':5,'never':0, 'unknown':0}})
df['last_new_job'] = pd.to_numeric(df['last_new_job'])
df['city'] = df['city'].str.replace('city_', '').astype(int)

gender_cat = pd.get_dummies(df['gender'], prefix='gender')
univer_cat = pd.get_dummies(df['enrolled_university'], prefix='univer')
edu_cat = pd.get_dummies(df['education_level'], prefix='edu')
major_cat = pd.get_dummies(df['major_discipline'], prefix='major')
comp_size_cat = pd.get_dummies(df['company_size'], prefix='comp_size')
comp_type_cat = pd.get_dummies(df['company_type'], prefix='comp_type')

#print(comp_type_cat.head())

df.drop(['gender','enrolled_university','education_level','major_discipline','company_size','company_type'], axis=1,inplace=True)
df = pd.concat((df, gender_cat, univer_cat, edu_cat, major_cat, comp_size_cat, comp_type_cat), axis=1)

# Normalization to values between 0 and 1 from here https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=df.columns)

# Makes bottom 30% of dataset the test and validation dataset
train, test_validation = train_test_split(df, test_size=0.3)
test, validation = train_test_split(test_validation, test_size=0.5)

# Storing the output that will be evaluated against later
output_train  = train.pop('target')
output_test = test.pop('target')
output_validation = validation.pop('target')

# Neural Network architecture
model = Sequential()
model.add(Dense(4, input_dim=len(train.columns), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))

print(model.summary())

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# Define callbacks
callback_a = ModelCheckpoint(filepath='my_best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)
callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

# Train the model and store the output into history variable
history = model.fit(train, output_train, validation_data=(test, output_test), epochs=13, batch_size=20, callbacks=[callback_a, callback_b])

# The below will help with visualizing the learning curve to see how the adjusted variables effect it
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.ylabel('Accuracy')
#plt.xlabel('epoch')
#plt.legend(['training data', 'validation data'], loc='lower right')
#plt.show()

# Load best model generated, this is to be used when running the program multiple times
model.load_weights('my_best_model.hdf5')
scores = model.evaluate(train, output_train)
print('training data')
print(model.metrics_names)
print(scores)

# Evaluate against validation dataset
scores = model.evaluate(validation, output_validation)
print('validation data')
print(model.metrics_names)
print(scores)

prediction = model.predict(validation)
print(output_validation.head(10))
print(prediction[0:10].round())

plt.plot(output_test, prediction, '.', alpha=0.3)
plt.xlabel('Correct labels')
plt.ylabel('Predicted confidence scores')

#  I dont like what this is showing. It doesnt show the trends I would expect and makes me think the datas distribution is leading to the 78% accuracy
plt.show()