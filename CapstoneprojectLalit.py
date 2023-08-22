#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
car_details = pd.read_csv('C:/Users/dell/Desktop/capstone/CAR DETAILS.csv')

#Exploring data and Performing Data Cleaning and Pre-Processing
car_details.head() #Showing the first five rows
car_details.shape # Showing the shape of the Dataset
car_details.dtypes # Checking Datatypes of all column
car_details.info() # Check ratings info
car_details.duplicated().sum() # Check Duplicates
car_details.columns #show all columms
car_details.isnull().sum() # Check the presence of missing values
import datetime
date_time = datetime.datetime.now()
car_details['Age']=date_time.year - car_details['year']
car_details.drop('year',axis=1,inplace=True)

car_details['name'].unique() # Showing Unique values in 'name' column
car_details['fuel'].unique() # Showing Unique values in 'fuel' column
car_details['seller_type'].unique() # Showing Unique values in 'seller_type' column
car_details['transmission'].unique() # Showing Unique values in 'transmission' column
car_details['owner'].unique() # Showing Unique values in 'owner' column

car_details['fuel']=car_details['fuel'].map({'Petrol':0,'Diesel':1, 'CNG':2, 'LPG':3, 'Electric':4})
car_details['fuel'].unique()
car_details['seller_type']=car_details['seller_type'].map({'Individual':0, 'Dealer':1, 'Trustmark Dealer':2})
car_details['seller_type'].unique()
car_details['transmission']=car_details['transmission'].map({'Manual':0, 'Automatic':1})
car_details['transmission'].unique()
car_details['owner']=car_details['owner'].map({'First Owner':0,'Second Owner':1,'Fourth & Above Owner':3, 'Third Owner':2, 'Test Drive Car':4})
car_details['owner'].unique()

import seaborn as sns
sns.boxplot(car_details['selling_price']) #check outlier value
sorted(car_details['selling_price'],reverse=True) # check car details
car_details = car_details[~(car_details['selling_price']>=8150000) & (car_details['selling_price']<=8900000)] # (~) skip those rows which are mention here
car_details.shape

#One-Hot Encoding for categorical variables
# Perform One-Hot Encoding
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
car_details_encoded = pd.get_dummies(car_details, columns=categorical_columns, drop_first=True)
#Imputation
# Example of imputing missing values with the mean
car_details_encoded['km_driven'].fillna(car_details_encoded['km_driven'].mean(), inplace=True)
from sklearn.preprocessing import StandardScaler
#Scaling of Data
# Example of scaling numerical columns using StandardScaler
numerical_columns = ['Age', 'selling_price', 'km_driven']
scaler = StandardScaler()
car_details_encoded[numerical_columns] = scaler.fit_transform(car_details_encoded[numerical_columns])
# Bar plot for fuel types
sns.countplot(x='fuel', data=car_details)
plt.title('Count of Different Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

# Bar plot for fuel types
sns.countplot(x='Age', data=car_details)
plt.title('Count of Different Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# Bar plot for seller types
sns.countplot(x='seller_type', data=car_details)
plt.title('Count of Different Seller Types')
plt.xlabel('Seller Type')
plt.ylabel('Count')
plt.show()

# Bar plot for transmission types
sns.countplot(x='transmission', data=car_details)
plt.title('Count of Different Transmission Types')
plt.xlabel('Transmission Type')
plt.ylabel('Count')
plt.show()

# Bar plot for owner types
sns.countplot(x='owner', data=car_details)
plt.title('Count of Different Owner Types')
plt.xlabel('Owner Type')
plt.ylabel('Count')
plt.show()

# Scatter plot for selling price vs. year
sns.scatterplot(x='Age', y='selling_price', data=car_details)
plt.title('Selling Price vs. Year')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.show()

# Scatter plot for km driven vs. selling price
sns.scatterplot(x='km_driven', y='selling_price', data=car_details)
plt.title('Selling Price vs. Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.show()

# Box plot for selling price distribution by fuel type
sns.boxplot(x='fuel', y='selling_price', data=car_details)
plt.title('Selling Price Distribution by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.show()

sns.boxplot(car_details['selling_price'])

# Extract the brand information from the 'name' column
car_details['Age'] = car_details['name'].str.split(' ', n=1).str[0]

# Delete the 'name' column
car_details.drop(columns=['name'], inplace=True)

# Convert categorical variables into numerical representations using one-hot encoding
car_details = pd.get_dummies(car_details, columns=['fuel', 'seller_type', 'transmission', 'owner'])

# Split the dataset into features (X) and the target variable (y)
X = car_details.drop(columns=['selling_price'])
y = car_details['selling_price']

print(car_details)
print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
# Encode categorical variables
label_encoder = LabelEncoder()
car_details['fuel'] = label_encoder.fit_transform(car_details['fuel_0'])
car_details['seller_type'] = label_encoder.fit_transform(car_details['seller_type_0'])
car_details['transmission'] = label_encoder.fit_transform(car_details['transmission_0'])
car_details['owner'] = label_encoder.fit_transform(car_details['owner_0'])
car_details['Age'] = label_encoder.fit_transform(car_details['Age'])

# Separate features (X) and target (y)
X = car_details.drop(columns=['selling_price'])
y = car_details['selling_price']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
# Regression Model (Linear Regression)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict on the test set
reg_pred = reg_model.predict(X_test)

from sklearn.metrics import mean_squared_error
# Evaluation Metrics for Regression (Root Mean Squared Error)
reg_rmse = mean_squared_error(y_test, reg_pred, squared=False)

# Print the evaluation result
print("Regression Model - Root Mean Squared Error:", reg_rmse)

from sklearn.ensemble import RandomForestClassifier
# Classification Model (Random Forest)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)

from sklearn.ensemble import BaggingClassifier
# Ensemble Model (Bagging)
bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100),
                                  n_estimators=10,
                                  random_state=42)
bagging_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
# Predict on the test set
reg_pred = reg_model.predict(X_test)
clf_pred = clf_model.predict(X_test)
bagging_pred = bagging_model.predict(X_test)

# Evaluation Metrics for Regression (Root Mean Squared Error)
reg_rmse = mean_squared_error(y_test, reg_pred, squared=False)

# Evaluation Metrics for Classification (Accuracy)
clf_accuracy = accuracy_score(y_test, clf_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)

# Print the evaluation results
print("Regression Model - Root Mean Squared Error:", reg_rmse)
print("Classification Model (Random Forest) - Accuracy:", clf_accuracy)
print("Ensemble Model (Bagging) - Accuracy:", bagging_accuracy)

#Save the best model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming X contains the features (excluding 'selling_price') and y contains the target ('selling_price')
X = car_details.drop(columns=['selling_price'])
y = car_details['selling_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE) as an evaluation metric
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the model to a file
import joblib
joblib.dump(model, 'best_model.pkl')

#Train a machine learning model on the original dataset.
from sklearn.linear_model import LinearRegression

# Split the dataset into features (X) and target (y)
X = car_details.drop(columns=['selling_price'])
y = car_details['selling_price']

# Initialize the model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

import random
# Create a new dataset by randomly selecting 20 data points
random.seed(42)  # Set a seed for reproducibility
random_indices = random.sample(range(len(car_details)), 20)
new_dataset = car_details.iloc[random_indices]
# Display the new dataset
print(new_dataset)

#Test the trained model on the new dataset.
# Split the new dataset into features (X_new) and target (y_new)
X_new = new_dataset.drop(columns=['selling_price'])
y_new_actual = new_dataset['selling_price']

# Use the trained model to predict on the new dataset
y_new_pred = model.predict(X_new)

# Compare the actual and predicted values
results = pd.DataFrame({'Actual Selling Price': y_new_actual, 'Predicted Selling Price': y_new_pred})
print(results)


