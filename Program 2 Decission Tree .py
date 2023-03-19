import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# read the data
data = pd.read_csv(r'C:\Users\bipro\OneDrive\Documents\Machine Learning\practical\weather.csv')
data.head()
data.info()

#clean the data
data_2 = data.dropna()
data_2.info()

# defining high humid as "Humidity3pm>28" at 3pm
data_2['high_humidity_label'] = (data_2['Humidity3pm']>28)*1

# create the target variable data for modelling
y = data_2[['high_humidity_label']].copy()

# create a list of all independent features apart from "Humidity3pm"
morning_features = ['WindSpeed9am', 'Humidity9am', 'Pressure9am','Cloud9am','Temp9am']

x = data_2[morning_features].copy()

# Model training
#Splitting data into train test split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state =324)

# Fit the decision tree model on data
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=15, random_state=0)
humidity_classifier.fit(X_train, y_train)

#predict value on train test split
y_predicted = humidity_classifier.predict(X_test)

#check the test accuracy
accuracy_score(y_test, y_predicted)*100

confusion_matrix(y_test, y_predicted)