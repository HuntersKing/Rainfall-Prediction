import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle

# Load the dataset to pandas dataframe.
data = pd.read_csv("Rainfall.csv")
print(type(data))


#just to check the rows and columns of dataset.
#data.shape

# print 5 rows from the starting and 5 rows from ending.
#print(data.head())
#print(data.tail())

data["day"].unique()

#Printing ths Data Info
print("Data Info:")
print(data.info())

# checking no of coloumns
data.columns

#remo extra spaces in all columns
data.columns = data.columns.str.strip()

print("Data Info:")
print(data.info())

# removing days as we don't need this
data = data.drop(columns=["day"])

## checking the number of missing values
print(data.isnull().sum())

data["winddirection"].unique()

 #handle the missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0]) 
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())  

data['rainfall'].unique()

# converting the yes & no in 1 & 0 form
data["rainfall"] = data["rainfall"].map({"yes":1,"no":0})
data["rainfall"].unique()

data.shape

# setting plot style for all plots
sns.set(style="whitegrid")
data.describe()

data.columns

#pLOTING THE GRAPH
plt.figure(figsize=(15,10))
for i ,columns in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'windspeed'],1):
  plt.subplot(3,3,i)
  sns.histplot(data[columns],kde=True)
  plt.title(f"Distribution of {columns}") 

plt.tight_layout()
plt.show()


#Ploting the graph for Rainfall Column
plt.figure(figsize=(6,4))
sns.countplot(x='rainfall',data=data)
plt.title("Count of Rainfall")
plt.show()


# correlation matrix( how each column s are correlate to other columns)
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation HeatMap")
plt.show()


plt.figure(figsize=(15,10))
for i ,columns in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'windspeed'],1):
  plt.subplot(3,3,i)
  sns.boxplot(data[columns])
  plt.title(f"Boxplot of {columns}") 

plt.tight_layout()
plt.show()

#Droping highly correlated columns
data=data.drop(columns=["maxtemp","mintemp","temparature"])

data.head()

# Down-sampling
print(data['rainfall'].value_counts())
# separate  majority and minority class
df_majority = data[data['rainfall']==1]
df_minority = data[data['rainfall']==0]
print(df_majority.shape)
print(df_minority.shape)

# down sample majority claass to match minority count
# resample
df_majority_downsampled = resample(df_majority,replace=False,n_samples=len(df_minority), random_state=42)
df_majority_downsampled.shape

df_downsampled = pd.concat([df_majority_downsampled,df_minority])

df_downsampled.shape 
df_downsampled.head()

# shuffling the final dataframe
df_downsampled = df_downsampled.sample(frac=1,random_state=42).reset_index(drop=True)

# split the data into training datasets and testing datasets
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Modal Training
rf_model = RandomForestClassifier(random_state=42) 

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30,None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#HyperTuning using GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("Best parameters for Random Forest Model:",grid_search_rf.best_params_)

cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Test dataset Performance
y_pred = best_rf_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test set confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("Test set classification report:")
print(classification_report(y_test,y_pred))

input_data = [1015.9,19.9,95,81,0,40,13.7]
input_df = pd.DataFrame([input_data],columns=X.columns)
prediction = best_rf_model.predict(input_df)
print("Prediction Result:","Rainfall" if prediction[0]==1 else "No Rainfall")

#prediction
# Save Model and Fearture name to a pickle file
model_data={"model":best_rf_model,"features":list(X.columns)}
pickle.dump(model_data,open("model.pkl","wb"))


#Using Loaded Pickle file for the prediction

import pickle
import pandas as pd
# load the trained model and features names from the pickle file
with open("model.pkl","rb") as file:
  model_data = pickle.load(file)

  model = model_data["model"]
features_names = model_data["features"]


input_data = [1015.9,19.9,95,81,0,40,13.7]
input_df = pd.DataFrame([input_data],columns=features_names)

prediction = best_rf_model.predict(input_df)
print("Prediction Result:","Rainfall" if prediction[0]==1 else "No Rainfall")