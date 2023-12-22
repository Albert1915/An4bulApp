# %%
#Connecting Colab with My Drive
from google.colab import drive
drive.mount('/content/gdrive')

# %% [markdown]
# ##1. IMPORT LIBRARIES

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop

# %% [markdown]
# ##2. CALL DATASET

# %%
#let's call our dataset
breed_dataset = pd.read_csv('/content/gdrive/MyDrive/Breed_Traits.csv')
display(breed_dataset)

# %% [markdown]
# We don't need to scale the data because there is no interval or ratio feature

# %%
#Replacing Categorical into Numeric
#For Coat Length
breed_dataset['Coat Length'].replace(['Short','Medium','Long'],
                                     [1,2,3], inplace=True)

#For Coat Type
breed_dataset['Coat Type'].replace(['Corded','Curly','Double','Hairless','Rough','Silky','Smooth','Wavy','Wiry'],
                                   [0,1,2,3,4,5,6,7,8], inplace=True)
display(breed_dataset)

# %%
#Let's remove some feature with PCA
x = breed_dataset.drop(['Breed','user_id'], axis=1)
reduce_feature = PCA()
reduce_feature.fit(x)

# %%
#Plotting How Many Features should we keep
plt.figure(figsize = (10,8))
plt.plot(range(1,17), reduce_feature.explained_variance_ratio_.cumsum(), marker = 'o')
plt.title('Explained Variance by Components')
plt.xlabel('n_components')
plt.ylabel('Cumulative Explained Variances')
plt.show()

#From the plot, keep 9 components from 16 components

# %%
#Reducing the components
reduce_feature = PCA(n_components = 9)
reduce_feature.fit(x)
x_transform = reduce_feature.transform(x)

# %%
print(x_transform)

# %% [markdown]
# ## 3. K-Means Analysis

# %% [markdown]
# 

# %%
#Let's try many clustering
Cluster_2 = KMeans(n_clusters=2, random_state=42)
Cluster_3 = KMeans(n_clusters=3, random_state=42)
Cluster_4 = KMeans(n_clusters=4, random_state=42)
Cluster_5 = KMeans(n_clusters=5, random_state=42)
Cluster_6 = KMeans(n_clusters=6, random_state=42)
Cluster_7 = KMeans(n_clusters=7, random_state=42)
Cluster_8 = KMeans(n_clusters=8, random_state=42)
Cluster_9 = KMeans(n_clusters=9, random_state=42)
Cluster_10 = KMeans(n_clusters=10, random_state=42)
Cluster_11 = KMeans(n_clusters=11, random_state=42)
Cluster_12 = KMeans(n_clusters=12, random_state=42)
Cluster_13 = KMeans(n_clusters=13, random_state=42)
Cluster_14 = KMeans(n_clusters=14, random_state=42)
Cluster_15 = KMeans(n_clusters=15, random_state=42)

# %%
#Evaluating the Cluster Model

inertia_metrics = []
index = range(2,16)
for i in index :
  kmeans = KMeans(n_clusters=i, random_state=42)
  kmeans.fit(x_transform)
  inertia_metrics_ = kmeans.inertia_
  inertia_metrics.append(inertia_metrics_)
  print(i, inertia_metrics_)

# %%
#Plot The Evaluation
plt.plot(index,inertia_metrics, marker = 'o', linestyle = '--')
plt.xlabel('n_cluster')
plt.ylabel('Sum Squared Error')
plt.show()

#According to plot, 6-8 Cluster will be enough.

# %%
#Fit the cluster 6
Cluster_6.fit(x_transform)

# %%
#Input The Cluster as Label
breed_dataset['Cluster'] = Cluster_6.labels_
display(breed_dataset)

# %% [markdown]
# ##4. Deep Learning for Recommendation

# %%
#Preparing dataset into Neural Network Shape
dataset_nn = breed_dataset.drop(['Breed','user_id'],axis=1)
y = dataset_nn['Cluster'].to_numpy().reshape(-1,1)
X = dataset_nn.drop(['Cluster'], axis=1).to_numpy().reshape(-1,16)


# %%
X_train.shape

# %%
y_train.shape

# %%
#Labeling y with 6 categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

dummy_y = to_categorical(encoded_Y)

# %%
dummy_y

# %%
#Splitting Dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,dummy_y, random_state = 30, train_size =0.8)

# %%
#It's Designing Architecture Time!
num_outputs = 6
tf.random.set_seed(42)
n_features = X_train.shape[1]

model_NN = Sequential()
model_NN.add(Dense(36, activation ='relu',kernel_initializer='he_normal', input_shape=(n_features,)))
model_NN.add(Dense(6, activation='softmax',kernel_initializer='he_normal'))

model_NN.compile(loss = 'categorical_crossentropy',
                 optimizer = Adam(learning_rate = 0.01),
                 metrics=['accuracy'])

# %%
#Early Stopping and Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(patience = 10, min_delta = 0.001, monitor = "val_accuracy")
checkpoint = ModelCheckpoint('Pet Recommendation.h5', verbose = 1, save_best_only = True)
callbacks_list = [early_stopping,checkpoint]

# %%
#Train the Model
history_train = model_NN.fit(X_train, y_train, epochs=1000, callbacks = callbacks_list,
                             batch_size=24,validation_data = (X_test,y_test))

# %%
#Plotting Model Accuracy
plt.plot(history_train.history['accuracy'])
plt.plot(history_train.history['val_accuracy'])
plt.title('Model Training and Testing Process')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

# %%
#Make prediction based on best model
from tensorflow.keras.models import load_model
best_model = load_model('Pet Recommendation.h5')
prediction = best_model.predict(X_test)
prediction_class = np.around(prediction)

# %%
prediction_class

# %%
dummy_y

# %% [markdown]
# Conclusion : So, if the user input some data in the system. The system will predict which breed cluster may the user interest


