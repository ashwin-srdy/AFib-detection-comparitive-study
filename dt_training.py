import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
# read in csv
df = pd.read_csv('data/training_25_features.csv')
print('-->DATA LOADED')
# train-test split
for i in range(23):
    df[df.columns[i]] = df[df.columns[i]].astype('float64')
X = df.drop(columns='ritmi')
y = df['ritmi']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.25, random_state = 1234)
print('-->DATA SPLIT DONE')
print('-->MODEL TRAINING')
parameters = { 
    'criterion':['gini','entropy'],
    'max_depth': np.arange(3, 15)
}
fitmodel = GridSearchCV(
    DecisionTreeClassifier(), 
    param_grid=parameters, 
    cv=5, 
    refit=True, scoring="accuracy", 
    n_jobs=-1, 
    verbose=1
)
fitmodel.fit(X_train, y_train)
import pickle as pkl
print('-->SAVING MODEL')
pkl.dump(fitmodel, open('ECG&patient_DT.pkl', 'wb'))
print('-->MODEL SAVED')
print('-->Testing model on test dataset')
y_pred = fitmodel.predict(X_test)
print(classification_report(y_test, y_pred))