import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r'C:\Users\bipro\OneDrive\Documents\Machine Learning\practical\weather.csv')
df_2 = df.dropna()

df_2['WindDir9am']=pd.factorize(df_2['WindDir9am'])[0]

df_2.columns

morning_features = ['WindDir9am','WindSpeed9am','Humidity9am','Pressure9am', 'Cloud9am', 'Temp9am']

df_2['high_humidity'] = (df_2['Humidity3pm']>28)*1

y = df_2[['high_humidity']].copy()

x = df_2[morning_features].copy()

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=324)

humidity_classifier = DecisionTreeClassifier(max_depth=20,max_leaf_nodes=15, random_state=0)

humidity_classifier.fit(X_train, y_train)

y_predict = humidity_classifier.predict(X_test)

accuracy_score(y_test,y_predict)*100

confusion_matrix(y_test, y_predict)
