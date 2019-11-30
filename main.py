# data analysis and modifying
import pandas as pd
import numpy as np
import random
# visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
from tensorflow import keras

# evaluation
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score


# Import DataFrame
beers_df = pd.read_csv('Data/beers.csv')
# Delete useless columns
beers_df = beers_df.drop(['UserId', 'Name'], axis=1)
# print(beers_df.info())
# PitchRate column has less than half filled records, so I delete it
beers_df = beers_df.drop(['PitchRate'], axis=1)

# splitting data into train and test sets
Y = beers_df['Style'].values
beers_df = beers_df.drop(['Style'], axis=1)
X = beers_df.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

# one hot encode categorials
categoricals = list(['Style'])
encoder = OneHotEncoder(sparse=False)
encoded_train = encoder.fit_transform(y_train.reshape(-1, 1))
encoded_test = encoder.fit_transform(y_test.reshape(-1, 1))

y_train_encoded = pd.DataFrame(encoded_train, columns=np.hstack(encoder.categories_))
y_test_encoded = pd.DataFrame(encoded_test, columns=np.hstack(encoder.categories_))

# my way to get rid of outliers: turn their huge values into max bound ones
X_train = pd.DataFrame(X_train, columns=beers_df.columns.values)
X_train.loc[X_train["ABV"] > 24., ["ABV"]] = 24.
X_train.loc[X_train["IBU"] > 500., ["IBU"]] = 500.
X_train.loc[X_train["Color"] > 40., ["Color"]] = 40.
X_train.loc[X_train["OG"] > 1.5, ["OG"]] = 1.5
X_train.loc[X_train["FG"] > 1.16, ["FG"]] = 1.16

X_test = pd.DataFrame(X_test, columns=beers_df.columns.values)
X_test.loc[X_test["ABV"] > 24., ["ABV"]] = 24.
X_test.loc[X_test["IBU"] > 500. , ["IBU"]] = 500.
X_test.loc[X_test["Color"] > 40, ["Color"]] = 40.
X_test.loc[X_test["OG"] > 1.5, ["OG"]] = 1.5
X_test.loc[X_test["FG"] > 1.16, ["FG"]] = 1.16

# fill missing rows from BoilGravity colums with mean value from X_train set
boil_gravity_mean = np.mean(X_train['BoilGravity'])
boil_gravity_median = np.median(X_train['BoilGravity'])
X_train['BoilGravity'] = X_train['BoilGravity'].fillna(boil_gravity_mean)

# same here but ATTENTION: use mean val from train set not test set!!!
X_test = pd.DataFrame(X_test, columns=beers_df.columns.values)
X_test['BoilGravity'] = X_test['BoilGravity'].fillna(boil_gravity_mean)

# standard scaling data - scaler is fitted with train set values only
scaler_train = StandardScaler().fit(X_train)
X_train = scaler_train.transform(X_train)
X_test = scaler_train.transform(X_test)

# make it DataFrame object again
X_train = pd.DataFrame(X_train, columns=beers_df.columns.values)
X_test = pd.DataFrame(X_test, columns=beers_df.columns.values)



# building Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='softmax'))
# load existing model from *.h5 file
# model = keras.models.load_model('./beers_model_pretrained_1.h5')

# compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[keras.metrics.categorical_accuracy])
history = model.fit(X_train, y_train_encoded, batch_size=512, epochs=1024, callbacks = [keras.callbacks.TerminateOnNaN()])
# print(history) # it is better to use jupyter notebook to see history


# evaluate model
score_test = model.evaluate(X_test, y_test_encoded, batch_size=1024)
# print(score_test)

score_train = model.evaluate(X_train, y_train_encoded, batch_size=1024)
# print(score_train)


# import Kaggle's submission test dataset
beers_test_nostyle_df = pd.read_csv('Data/beers_test_nostyle.csv')
# do same transformation on this dataset
# outliers
beers_test_nostyle_df = beers_test_nostyle_df.drop(['UserId', 'PitchRate', 'Id'], axis=1)
beers_test_nostyle_df.loc[beers_test_nostyle_df["ABV"] > 24., ["ABV"]] = 24.
beers_test_nostyle_df.loc[beers_test_nostyle_df["IBU"] > 500., ["IBU"]] = 500.
beers_test_nostyle_df.loc[beers_test_nostyle_df["Color"] > 40., ["Color"]] = 40.
beers_test_nostyle_df.loc[beers_test_nostyle_df["OG"] > 1.5, ["OG"]] = 1.5
beers_test_nostyle_df.loc[beers_test_nostyle_df["FG"] > 1.16, ["FG"]] = 1.16
# filling NAs
beers_test_nostyle_df['BoilGravity'] = beers_test_nostyle_df['BoilGravity'].fillna(boil_gravity_mean)
# do prediction
preds_sumbit = model.predict(beers_test_nostyle_df)

# convert predictions into *.csv file required in Kaggle's competition
preds_sumbit_array = [['Id', 'Style']]
for pred in range(len(preds_sumbit)):
    # convert each prediction from probability float into name string
    index = np.argmax(preds_sumbit[pred])
    preds_sumbit_array.append([pred,np.array(encoder.categories_).reshape(-1,1)[index][0]])

# save it into *.csv file
with open('my_submission_1.csv', 'w+') as file:
    for row in preds_sumbit_array:
            file.write('{0},{1}\n'.format(row[0],row[1]))

# save model into *.h5 file
model.save('beers_model_pretrained_1.h5')
