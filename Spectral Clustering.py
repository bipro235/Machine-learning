import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Loading the dataset

data = pd.read_csv(r'C:\Users\bipro\OneDrive\Documents\Machine Learning\practical\CC GENERAL (1).csv')

# Drop the "CUST_ID" from the dataset
data = data.drop('CUST_ID',axis=1)

# Handle the missing value if any
data.fillna(method="ffill",inplace=True)

# Preprocess the data to make them suitable for visualization
# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#Normalize the data
data_normalized = normalize(data_scaled)

#Convert the numpy into pandas data frame
data_normalized = pd.DataFrame(data_normalized)

#Reducing the dimensions of the data
pca = PCA(n_components=2)
data_principal = pca.fit_transform(data_normalized)
data_principal = pd.DataFrame(data_principal)
data_principal.columns = ['P1','P2']


# Build the clustering model and visualize the clustering model
spectral_model_rbf = SpectralClustering(n_clusters=2, affinity='rbf')

#Training the model and store the predicted clusters label
labels_rbf = spectral_model_rbf.fit_predict(data_principal)


#Build the label to colour mapping
colours ={}
colours[0] = 'b'
colours[1] = 'y'

# Building the colour vector for each data point
cvec = [colours[label] for label in labels_rbf]

#plotting the clustered scatter plot
b = plt.scatter(data_principal['P1'], data_principal['P2'], color ='b');
y = plt.scatter(data_principal['P1'], data_principal['P2'], color ='y');
 
plt.figure(figsize =(9, 9))
plt.scatter(data_principal['P1'], data_principal['P2'], c = cvec)
plt.legend((b, y), ('Label 0', 'Label 1'))
plt.show()






















