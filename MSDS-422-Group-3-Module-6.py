#!/usr/bin/env python
# coding: utf-8

# ## Appendix 1 - Python Code and Outputs

# ### Data Preparation

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### Import Training Data

# In[2]:


import numpy as np
import pandas as pd
# load training data
digit_training_data = pd.read_csv('train.csv')

# show first rows of the data
digit_training_data.head(100)
# show number of columns and rows
digit_training_data.shape


# ### Investigation of Missing Data and Outliers in Training Data

# In[3]:


# find null counts, percentage of null values, and column type
null_count = digit_training_data.isnull().sum()
null_percentage = digit_training_data.isnull().sum() * 100 / len(digit_training_data)
column_type = digit_training_data.dtypes

# show null counts, percentage of null values, and column type for columns with more than one Null value
null_summary = pd.concat([null_count, null_percentage, column_type], axis=1, keys=['Missing Count', 'Percentage Missing','Column Type'])
null_summary_only_missing = null_summary[null_count != 0].sort_values('Percentage Missing',ascending=False)
null_summary_only_missing


# The above analysis displays that there is no missing data in the digit recognizer training dataset.

# ### Import Testing Data

# In[4]:


# import test dataset
digit_testing_data = pd.read_csv('test.csv')

# show first ten rows of the data
digit_testing_data.head(10)
# show number of columns and rows
digit_testing_data.shape


# ### Investigation of Missing Data and Outliers in Training Data

# In[5]:


# find null counts, percentage of null values, and column type
null_count = digit_testing_data.isnull().sum()
null_percentage = digit_testing_data.isnull().sum() * 100 / len(digit_training_data)
column_type = digit_testing_data.dtypes

# show null counts, percentage of null values, and column type for columns with more than one Null value
null_summary = pd.concat([null_count, null_percentage, column_type], axis=1, keys=['Missing Count', 'Percentage Missing','Column Type'])
null_summary_only_missing = null_summary[null_count != 0].sort_values('Percentage Missing',ascending=False)
null_summary_only_missing


# The above analysis displays that there is no missing data in the digit recognizer test dataset.

# ### Apply Principal Components Analysis (PCA) to Combined Training and Test Data

# First, we will combine the training and test dataframes

# In[6]:


# Create a copy of the training dataframe
pca_train_df = digit_training_data.copy(deep=True)

# Drop the label column from the copy of the training dataframe
pca_train_df.drop(['label'], axis=1, inplace=True)

# Concatenate the training and test dataframes
pca_df = pd.concat([pca_train_df, digit_testing_data])

# show first rows of the data
pca_df.head(10)
# show number of columns and rows
pca_df.shape
# Describe the dataframe
pca_df.describe()


# find null counts, percentage of null values, and column type
null_count = pca_df.isnull().sum()
null_percentage = pca_df.isnull().sum() * 100 / len(digit_training_data)
column_type = pca_df.dtypes

# show null counts, percentage of null values, and column type for columns with more than one Null value
null_summary = pd.concat([null_count, null_percentage, column_type], axis=1, keys=['Missing Count', 'Percentage Missing','Column Type'])
null_summary_only_missing = null_summary[null_count != 0].sort_values('Percentage Missing',ascending=False)
null_summary_only_missing


# ### Construct a Random Forest Model Using the full training model

# First let's load the required packages:

# In[10]:


#Import required Modules
#pip install graphviz

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


# Next, the training and validation datasets were utilized to conduct hyperparameter tuning to find the best hyperparameters for random forest modeling.

# In[11]:


# Start a timer for the Random Forest
rf_start = datetime.datetime.now()

# Import Required Modules
#pip install graphviz
#import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Create a copy of the training dataframe
rf_train_df = digit_training_data.copy(deep=True)

# Drop the label column from the copy of the training dataframe
rf_train_df.drop(['label'], axis=1, inplace=True)


# Split the training dataset into predictor and outcome components
rf_train_x = rf_train_df
rf_train_y = digit_training_data['label']

# Split the Kaggle training data into training and validation components
rf_x_train, rf_x_validation, rf_y_train, rf_y_validation = train_test_split(rf_train_x,
                                                                      rf_train_y, 
                                                                            test_size=0.25, 
                                                                           random_state = 1)

# Conduct hyperparameter tuning for random forest models
param_dist = {'n_estimators': randint(10,100),
              'max_depth': randint(1,100),
             'max_features': randint(1,20)}

rf = RandomForestClassifier()

#This approach uses 5-fold cross-validation
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(rf_train_x, rf_train_y)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Print the run time for Python to complete the Random Forest
rf_end = datetime.datetime.now()
rf_runtime = rf_end - rf_start
print(f"The total run time for the Random Forest Model using the training dataset was {rf_runtime}.")


# Next, let's examine the strength of the random forest model associated with the optimal hyperparameters by applying the model to the validation dataset and examining the resulting confusion matrix, accuracy, precision, and recall.

# In[12]:


# Generate predictions with the best model
y_predictions_rf = best_rf.predict(rf_x_validation)

# Create the confusion matrix associated with the best random forest model
cm = confusion_matrix(rf_y_validation, y_predictions_rf)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy associated with the predictions of the best random forest model

accuracy_rf_validation = accuracy_score(rf_y_validation, y_predictions_rf)

print("Accuracy:", accuracy_rf_validation)


# In[13]:


from sklearn.metrics import classification_report
# print classification report 
print(classification_report(rf_y_validation, y_predictions_rf)) 


# Apply the Random Forest Model to the Test Dataframe

# In[14]:


# Create a copy of the training dataframe
rf_testing_x = digit_testing_data.copy(deep=True)

# Drop the label column from the copy of the training dataframe
#rf_testing_x.drop(['label'], axis=1, inplace=True)

# Apply the Random Forest model to the test dataset
y_test_predictions_rf = best_rf.predict(rf_testing_x)

# Put the random forest predictions into a Pandas dataframe
prediction_df_rf = pd.DataFrame(y_test_predictions_rf, columns=['Label'])

# Add the ID column to the front of the random forest predictions dataframe
ImageId_series = pd.Series(range(1,28001))
prediction_df_rf.insert(0, 'ImageId', ImageId_series)

#output predictions to csv
#prediction_df_rf.to_csv('test_predictions_rf1.csv', index=False)


# In[15]:


import matplotlib.pyplot as plt
# Display the kaggle results associated with the Random Forest Model
plt.figure(figsize = (15, 15))
kaggle_results = plt.imread('Digit_Random_Forest1_Kaggle_Results_v1.png')
plt.imshow(kaggle_results)
plt.axis("off")
plt.show()


# # Next, we scale the data to prepare it for our principal components analysis

# In[7]:


# Scale PCA dataframe's data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
pca_scaled = sc.fit_transform(pca_df) # normalizing the features

# Convert scaled data from numpy array into dataframe
pca_features = list(pca_df.columns.values)
pca_scaled_df = pd.DataFrame(pca_scaled, columns=pca_features)

# Confirm scaling transformation was a success
pca_scaled_df.shape
pca_scaled_df.head(10)
pca_scaled_df.describe()


# We also apply this scaling to our test dataframe for later use as we progress through the construction of our Principal Component Analysis and Random Forest model creation processes.

# In[8]:


# Apply the standard scaling to the test dataframe
pca_test_scaled = sc.transform(digit_testing_data)

# Convert scaled data from numpy array into dataframe
pca_test_features = list(digit_testing_data.columns.values)
pca_test_scaled_df = pd.DataFrame(pca_test_scaled, columns=pca_test_features)

# Confirm scaling transformation was a success
pca_test_scaled_df.shape
pca_test_scaled_df.head(10)
pca_test_scaled_df.describe()


# Next, we will conduct a Principal Components Analysis to identify principal components that account for at least 95% of the variation in the data.

# In[9]:


# Start a timer for the Principal Components Analysis
import datetime
pca_start = datetime.datetime.now()

# Applying PCA function on training and testing set of X component
from sklearn.decomposition import PCA
pca_digits_train_test = PCA(n_components=334)
principal_components_digits = pca_digits_train_test.fit_transform(pca_scaled_df)


# Create a Cumulative Scree plot to help us determine how many principal components to include in our random forest model
import matplotlib.pyplot as plt
import numpy as np

PC_values = np.arange(pca_digits_train_test.n_components_) + 1
cumulative_explained_variance_pca = np.cumsum(pca_digits_train_test.explained_variance_ratio_)

plt.plot(PC_values, cumulative_explained_variance_pca, 'o-', linewidth=1, color='blue')
plt.title('Cumulative Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.show()

# Create a dataframe to display the information in the cumulative scree plot in a different manner
scree_df = pd.DataFrame({'Principal Component':PC_values, 'Variance Explained':cumulative_explained_variance_pca})
scree_df

# Create a dataframe that contains the principal component values for each of the observations in the pca dataframe
pca_column_list = []
for num in range(1, 335):
    pca_column_list.append("PC_" + str(num))

pca_digits_df = pd.DataFrame(data = principal_components_digits , columns = pca_column_list )

pca_digits_df


# Print the run time for Python to complete the Principal Components Analysis
pca_end = datetime.datetime.now()
pca_runtime = pca_end - pca_start
print(f"The total run time for the Principal Components Analysis was {pca_runtime}.")


# ### Construct a Random Forest Model Using the Principal Components Identified

# Let's fit a Random Forest Model to predict digits using the principal components just identified.  We will use our training and validation datasets to conduct hyperparameter tuning to find the best hyperparameters for random forest modeling.

# In[16]:


# Start a timer for the Random Forest

pca_rf_start = datetime.datetime.now()

# Create the Random Forest Model

# Import Required Modules
#pip install graphviz
#import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Split the training dataset into predictor and outcome components
rf_train_validation_x = pca_digits_df.copy(deep=True)
rf_train_validation_x.drop(rf_train_validation_x.tail(28000).index, inplace = True)
rf_train_validation_y = digit_training_data['label']



# Split the Kaggle training data into training and validation components
rf_x_train, rf_x_validation, rf_y_train, rf_y_validation = train_test_split(rf_train_validation_x,
                                                                      rf_train_validation_y, 
                                                                            test_size=0.2, 
                                                                           random_state = 1)

# Conduct hyperparameter tuning for random forest models
param_dist = {'n_estimators': randint(10,100),
              'max_depth': randint(1,100),
             'max_features': randint(1,20)}

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(rf_x_train, rf_y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Print the run time for Python to complete the Random Forest
pca_rf_end = datetime.datetime.now()
pca_rf_runtime = pca_rf_end - pca_rf_start
print(f"The total run time for the Random Forest Model using the principal components was {pca_rf_runtime}.")


# Next, we will assess the strength of the random forest model associated with the optimal hyperparameters by applying the model to the validation dataset and observing the resulting confusion matrix and accuracy.

# In[17]:



# Generate predictions with the best model
y_validation_predictions_rf = best_rf.predict(rf_x_validation)

# Create the confusion matrix associated with the best random forest model
cm = confusion_matrix(rf_y_validation, y_validation_predictions_rf)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy, precision, and recall associated with the predictions of the best random forest model

accuracy_rf_validation = accuracy_score(rf_y_validation, y_validation_predictions_rf)
#precision_rf_validation = precision_score(rf_y_validation, y_validation_predictions_rf)
#recall_rf_validation = recall_score(rf_y_validation, y_validation_predictions_rf)

print("Accuracy:", accuracy_rf_validation)
#print("Precision:", precision_rf_validation)
#print("Recall:", recall_rf_validation)


# Apply the Random Forest Model to the Test Dataframe

# In[18]:


# Create a dataframe for predictor variables in the test dataframe for random forest model
#rf_testing_x = rf_testing_df.drop(columns=['PassengerId'])
rf_testing_x = pca_digits_df.copy(deep=True)
rf_testing_x.drop(rf_testing_x.head(42000).index, inplace = True)

# Apply the Random Forest model to the test dataset
y_test_predictions_rf = best_rf.predict(rf_testing_x)

# Put the random forest predictions into a Pandas dataframe
prediction_df_rf = pd.DataFrame(y_test_predictions_rf, columns=['Label'])

# Add the ID column to the front of the random forest predictions dataframe
ImageId_series = pd.Series(range(1,28001))
prediction_df_rf.insert(0, 'ImageId', ImageId_series)

#output predictions to csv
#prediction_df_rf.to_csv('test_predictions_pca_random_forest_v1.csv', index=False)


# Let's display the Kaggle results from the application of the random forest model using principal components to the test dataset

# In[19]:


# Display the kaggle results associated with the Random Forest Model
plt.figure(figsize = (15, 15))
kaggle_results = plt.imread('Digit_PCA_Random_Forest_Kaggle_Results_v1.jpg')
plt.imshow(kaggle_results)
plt.axis("off")
plt.show()


# ### Construct a Random Forest Model Using the Principal Components Identified and the Original Data

# Let's fit a Random Forest Model to predict digits using the principal components and the original underlying data.  We will use our training and validation datasets to conduct hyperparameter tuning to find the best hyperparameters for random forest modeling.

# In[20]:


# Start a timer for the Random Forest

pca_rf_v2_start = datetime.datetime.now()



# Split the training dataset into predictor and outcome components
rf_train_validation_x = pca_digits_df.copy(deep=True)
rf_train_validation_x.drop(rf_train_validation_x.tail(28000).index, inplace = True)
rf_train_validation_x = pd.concat([rf_train_validation_x, pca_train_df], axis=1)
rf_train_validation_y = digit_training_data['label']

# Split the Kaggle training data into training and validation components
rf_x_train, rf_x_validation, rf_y_train, rf_y_validation = train_test_split(rf_train_validation_x,
                                                                      rf_train_validation_y, 
                                                                            test_size=0.2, 
                                                                           random_state = 1)

# Conduct hyperparameter tuning for random forest models
param_dist = {'n_estimators': randint(10,100),
              'max_depth': randint(1,100),
             'max_features': randint(1,20)}

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(rf_x_train, rf_y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)


# Print the run time for Python to complete the Random Forest
pca_rf_v2_end = datetime.datetime.now()
pca_rf_v2_runtime = pca_rf_v2_end - pca_rf_v2_start
print(f"The total run time for the Random Forest Model using the principal components and original pixel features was {pca_rf_v2_runtime}.")


# Next, we will assess the strength of the random forest model associated with the optimal hyperparameters by applying the model to the validation dataset and observing the resulting confusion matrix and accuracy.

# In[21]:


# Generate predictions with the best model
y_validation_predictions_rf = best_rf.predict(rf_x_validation)

# Create the confusion matrix associated with the best random forest model
cm = confusion_matrix(rf_y_validation, y_validation_predictions_rf)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy, precision, and recall associated with the predictions of the best random forest model

accuracy_rf_validation = accuracy_score(rf_y_validation, y_validation_predictions_rf)
#precision_rf_validation = precision_score(rf_y_validation, y_validation_predictions_rf)
#recall_rf_validation = recall_score(rf_y_validation, y_validation_predictions_rf)

print("Accuracy:", accuracy_rf_validation)
#print("Precision:", precision_rf_validation)
#print("Recall:", recall_rf_validation)


# Apply the Random Forest Model to the Test Dataframe

# In[22]:


# Create a dataframe for predictor variables in the test dataframe for random forest model
rf_testing_x = pca_digits_df.copy(deep=True)
rf_testing_x.drop(rf_testing_x.head(42000).index, inplace = True)
rf_testing_x.reset_index(drop=True, inplace=True)
digit_testing_data.reset_index(drop=True, inplace=True)
rf_testing_x = pd.concat([rf_testing_x, digit_testing_data], axis=1)

# Apply the Random Forest model to the test dataset
y_test_predictions_rf = best_rf.predict(rf_testing_x)

# Put the random forest predictions into a Pandas dataframe
prediction_df_rf = pd.DataFrame(y_test_predictions_rf, columns=['Label'])

# Add the ID column to the front of the random forest predictions dataframe
ImageId_series = pd.Series(range(1,28001))
prediction_df_rf.insert(0, 'ImageId', ImageId_series)

#output predictions to csv
#prediction_df_rf.to_csv('test_predictions_pca_random_forest_v2.csv', index=False)


# Let's display the Kaggle results from the application of the random forest model using principal components and the original underlying data features to the test dataset.

# In[23]:


# Display the kaggle results associated with the Random Forest Model
plt.figure(figsize = (15, 15))
kaggle_results = plt.imread('Digit_PCA_And_Original_Features_Random_Forest_Kaggle_Results_v1.jpg')
plt.imshow(kaggle_results)
plt.axis("off")
plt.show()


# In[24]:


# mitigate design flaw
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
train = digit_training_data.drop(columns = 'label')
train_label = digit_training_data['label']
scaled_train = sc.fit_transform(train)

pca = PCA(n_components=334)
pca_train = pca.fit_transform(scaled_train)

# Split the Kaggle training data into training and validation components
rf_x_train, rf_x_validation, rf_y_train, rf_y_validation = train_test_split(pca_train, train_label, test_size=0.2, random_state = 1)

rf = RandomForestClassifier()
rf.fit(rf_x_train, rf_y_train)
predictions = rf.predict(rf_x_validation)


# ### Deploy K-Means Clustering

# Let's use K-means clustering to predict digits using original features. First let's create our training and testing data and plot the digits in the dataset

# In[25]:


import sys
import sklearn
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Split the training dataset into predictor and outcome variables
kmeans_x_train = digit_training_data.copy(deep=True)
kmeans_x_train.drop(['label'], axis=1, inplace=True)
kmeans_y_train = digit_training_data['label']


kmeans_x_train = np.array(kmeans_x_train)
kmeans_y_train = np.array(kmeans_y_train)


print('Training Data: {}'.format(kmeans_x_train.shape))
print('Training Labels: {}'.format(kmeans_y_train.shape))

# reshape array to 3-dimensional array so we can plot the numbers
kmeans_x_train_plot = kmeans_x_train.reshape(42000, 28, 28)

# Plot the digits in the dataset
fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()

for i, ax in enumerate(axs.flat):
    ax.matshow(kmeans_x_train_plot[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(kmeans_y_train[i]))
    fig.show()


# Normalize the training data before applying k-means clustering

# In[26]:


from sklearn import preprocessing
kmeans_x_train_norm = preprocessing.normalize(kmeans_x_train)


# The MNIST dataset contains images of the integers 0 to 9. Because of this, let’s start by setting the number of clusters to 10, one for each digit

# Compute the silhouette coefficients kmeans models with different numbers of clusters. This can vary between –1 and +1. A coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster boundary; finally, a coefficient close to –1 means that the instance may have been assigned to the wrong cluster.
# 
# reference: Geron, Aurelien. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. 2nd ed. Sebastopol, CA: O'Reilly.

# In[27]:


import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import MiniBatchKMeans
# minibatchkmeans has a memory leak warning that we can ignore
import warnings
warnings.filterwarnings('ignore')

# create k-means models with K clusters. 
K = clusters=[10,16,36,64,144,256,400] # test listed cluster numbers

# Store within-cluster-sum of squares and silhouette scores for clusters
wss = []
sil_score = []

# loop though cluster values and save inertia and silhouttee values
for i in K:
    kmeans=MiniBatchKMeans(n_clusters=i, random_state=1)
    kmeans=kmeans.fit(kmeans_x_train_norm)
    # within-cluster-sum-squares
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    # silhouttee score
    score = silhouette_score(kmeans_x_train_norm, kmeans.labels_)
    sil_score.append(score)
    print ("Silhouette score for k(clusters) = "+str(i)+" is "+str(score))


# In[28]:


import seaborn as sns
# elbow and silhouttee scores in dataframe with number of clusters
cluster_sil_scores = pd.DataFrame({'Clusters' : K, 'WSS' : wss, 'Sil Score' : sil_score})
cluster_sil_scores

# plot the elbow scores
sns.lineplot(x = 'Clusters', y = 'WSS', data = cluster_sil_scores, marker="+")


# Based on the elbow plot, the inertia drops very quickly as we increase k up to 50, but then it decreases a bit more slowly as we keep increasing k. This curve has a distinct elbow shape, we also a more gradual decline around 250.
# 
# This indicates that 144 and 256 could be optimal cluster numbers.

# In[29]:


# plot the silhouttee scores
sns.lineplot(x = 'Clusters', y = 'Sil Score', data = cluster_sil_scores, marker="+")


# Based on the plot, silhouette scores decline as the number of clusters increases. Scores close to 0 suggest that the clusters are overlapping, and the model with more clusters may not able to distinguish them well.
# 
# This isn't what we observe with the inertia plot, so we will still test models with 144 and 256 clusters. We also know there are 10 digits that are represented in the dataset so this could also be an optimal cluster number. We will build three models using  these cluster numbers and compare performance metrics.

# K-means clustering is an unsupervised machine learning method so the labels assigned by our KMeans algorithm refer to the cluster each array was assigned to, not the actual target integer. This section defines functions that predict which integer corresponds to each cluster. reference: https://medium.datadriveninvestor.com/k-means-clustering-for-imagery-analysis-56c9976f16b6#:~:text=Preprocessing

# In[30]:


def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
  # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


# Let's build models with 10, 144, and 256 clusters based on our knowledge of the data and the elbow and silhouette plot analysis.

# In[31]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics

########### Initialize KMeans model with 10 clusters ##############
# Initialize KMeans model
kmeans = MiniBatchKMeans(n_clusters = 10, random_state=1)

# Fit the model to the training data
kmeans.fit(kmeans_x_train_norm)

# Predict the cluster assignment
X_clusters = kmeans.predict(kmeans_x_train_norm)
print(X_clusters[:20])

# predict labels for kmeans model with 10 clusters
cluster_labels=infer_cluster_labels(kmeans,kmeans_y_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)

# print first 20 predicted labels and actual y-values
print(predicted_labels[:20])
print(kmeans_y_train[:20])

# Create the confusion matrix
cm = confusion_matrix(kmeans_y_train, predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy, inertia, and homogeneity scores
accuracy_kmeans = accuracy_score(kmeans_y_train, predicted_labels)
inertia_kmeans = kmeans.inertia_
homogeneity_kmeans = metrics.homogeneity_score(kmeans_y_train,predicted_labels)
print("Accuracy of K=10:", accuracy_kmeans)
print("Inertia of K=10:", inertia_kmeans)
print("Homogeneity of K=10:", homogeneity_kmeans)

########### Initialize KMeans model with 144 clusters ##############
kmeans = MiniBatchKMeans(n_clusters = 144, random_state=1)

# Fit the model to the training data
kmeans.fit(kmeans_x_train_norm)

# Predict the cluster assignment
X_clusters = kmeans.predict(kmeans_x_train_norm)
print(X_clusters[:20])

# predict labels for kmeans model with 144 clusters
cluster_labels=infer_cluster_labels(kmeans,kmeans_y_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)

# print first 20 predicted labels and actual y-values
print(predicted_labels[:20])
print(kmeans_y_train[:20])

# Create the confusion matrix
cm = confusion_matrix(kmeans_y_train, predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy scores
accuracy_kmeans = accuracy_score(kmeans_y_train, predicted_labels)
inertia_kmeans = kmeans.inertia_
homogeneity_kmeans = metrics.homogeneity_score(kmeans_y_train,predicted_labels)
print("Accuracy of K=144:", accuracy_kmeans)
print("Inertia of K=144:", inertia_kmeans)
print("Homogeneity of K=144:", homogeneity_kmeans)


########### Initialize KMeans model with 256 clusters ##############
# Initialize KMeans model
kmeans = MiniBatchKMeans(n_clusters = 256, random_state=1)

# Fit the model to the training data
kmeans.fit(kmeans_x_train_norm)

# Predict the cluster assignment
X_clusters = kmeans.predict(kmeans_x_train_norm)
print(X_clusters[:20])

# predict labels for kmeans model with 256 clusters
cluster_labels = infer_cluster_labels(kmeans,kmeans_y_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)

# print first 20 predicted labels and actual y-values
print(predicted_labels[:20])
print(kmeans_y_train[:20])

# Create the confusion matrix
cm = confusion_matrix(kmeans_y_train, predicted_labels)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# Calculate the accuracy scores
accuracy_kmeans = accuracy_score(kmeans_y_train, predicted_labels)
inertia_kmeans = kmeans.inertia_
homogeneity_kmeans = metrics.homogeneity_score(kmeans_y_train,predicted_labels)
print("Accuracy of K=256:", accuracy_kmeans)
print("Inertia of K=256:", inertia_kmeans)
print("Homogeneity of K=256:", homogeneity_kmeans)


# We observe accuracy scores of 
#  - 0.594 for the k-means model with 10 clusters 
#  - 0.881 for the k-means model with 144 clusters
#  - 0.921 for the k-means model with 256 clusters.

# Visualizing Cluster Centroids
# 
# Let's display the most representative image for each cluster.

# In[32]:


# Initialize KMeans model with 256 clusters
kmeans = MiniBatchKMeans(n_clusters = 256, random_state=1)

# Fit the model to the training data
kmeans.fit(kmeans_x_train_norm)

# record centroid values
centroids = kmeans.cluster_centers_

# reshape centroids into images
images = centroids.reshape(256, 28, 28)
images *= 255
images = images.astype(np.uint8)

# determine cluster labels
cluster_labels = infer_cluster_labels(kmeans, kmeans_y_train)

# create figure with subplots using matplotlib.pyplot
fig, axs = plt.subplots(32, 8, figsize = (20, 20))
plt.gray();

# loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):
    
    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label:{}'.format(key), fontsize=8)
    
    # add image to subplot
    ax.matshow(images[i]);
    ax.axis('off');
    
# display the figure
fig.show();


# Apply the K-means Clustering Model to the Test Dataframe

# In[33]:


# Create a dataframe for predictor variables in the test dataframe for kmeans model
kmeans_testing_x = digit_testing_data.copy(deep=True)
#kmeans_testing_x.drop(['Label'], axis=1, inplace=True)

# Apply the kmeans model to the test dataset
y_test_prediction_clusters_kmeans = kmeans.predict(kmeans_testing_x)

# predict labels for kmeans model
kmeans_predictions = infer_data_labels(y_test_prediction_clusters_kmeans, cluster_labels)

# Put the kmeans predictions into a Pandas dataframe
prediction_df_kmeans = pd.DataFrame(kmeans_predictions, columns=['Label'])

# Add the ID column to the front of the kmeans predictions dataframe
ImageId_series = pd.Series(range(1,28001))
prediction_df_kmeans.insert(0, 'ImageId', ImageId_series)

# Output predictions to csv
#prediction_df_kmeans.to_csv('test_predictions_kmeans_v1.csv', index=False)


# Let's display the Kaggle results from the application of the kmeans model on the test dataset

# In[34]:


# Display the kaggle results associated with the Random Forest Model
plt.figure(figsize = (15, 15))
kaggle_results = plt.imread('Digit_Kmeans_v1.jpg')
plt.imshow(kaggle_results)
plt.axis("off")
plt.show()

