import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle


df = pd.read_csv('football_data.csv')

# Rename columns
df = df.rename(columns={'Unnamed: 0': 'League',
                        'Unnamed: 1': 'SeasonYear', 'missed': 'against'})


# # Dataframe without categorical variables and wins, draws, loses
df_model = df[['scored',
               'against', 'pts', 'xG', 'xGA', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed']]


# get dummy data (in models, each type of categorical variable needs their own column with corresponding 1 or 0 (yes or no))
# many people use one-hot-encoder, some think pandas get dummies is more effective
df_dum = pd.get_dummies(df_model)


# train test split - code taken from sklearn website (creates train set, validation set and test set)
X = df_dum.drop('pts', axis=1)
y = df_dum.pts.values
# y set as values form so that its in array form (recommended for use in models) (0.8 measn 80% train set 20% test set)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# multiple linear regression (sklearn)
# uses crossvalscore which takes sample from test and validations set, runs model and sees if model generalises well
lm = LinearRegression()
lm.fit(X_train, y_train)

lm_score_check = np.mean(cross_val_score(
    lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# lasso regression
lm_l = Lasso()
lm_l.fit(X_train, y_train)

lm_l_score_check = np.mean(cross_val_score(
    lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# random forest
rf = RandomForestRegressor()
rf_score_check = np.mean(cross_val_score(rf, X_train, y_train,
                                         scoring='neg_mean_absolute_error', cv=3))


# tune models using Gridserachcv
parameters = {'n_estimators': range(
    10, 300, 10), 'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

# print(gs.best_score_)
# print(gs.best_estimator_)


# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

print('mae lm = ', mean_absolute_error(y_test, tpred_lm))
print('mae lm_l = ', mean_absolute_error(y_test, tpred_lml))
print('mae rf = ', mean_absolute_error(y_test, tpred_rf))

# test combination of two models
print('mae lm + rf', mean_absolute_error(y_test, (tpred_lm+tpred_rf)/2))


# Productionise the model in a Flask API endpoint
# first pickle the model (converts the model into a byte stream which can be stored, transferred and converted back to the original model at a later time)

pickl = {'model': gs.best_estimator_}
pickle.dump(pickl, open('model_file' + '.p', 'wb'))

file_name = 'model_file.p'
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1, :].values.reshape(1, -1))


# Print an instance of the data to use as test data for API endpoint
print(X_test.iloc[1, :])
