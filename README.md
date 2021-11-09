# capstonePimaIndians
My final project for Ryerson University created to analyze the Pima Indians Diabetes dataset
I hope to be able to use the dataset to determine which factors are prevalent in determining diabetes, thus creating less of a strain on hospitals during the time of Covid
My data set will:
Find the correlation between the given attributes
Pick the attribute I want to focus on
Complete a heat map
And finally, use test-train models to compare accuracy 

#import all extenstions needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

df = pd.read_csv('diabetes.csv') #data loaded to Jupyter to view

df.head() #displaying the tables within the dataset

df.describe() #describing the data to see if any particuluar variable stands out, using the mean to help determine the biggest contributors

df.info()

df.isnull().sum() #Check data for any null values to ensure nothing is missing from set

df.isnull().all()

df.corr() #Find correlations of all the columns within the df

sns.set_style('darkgrid')
plt.figure(figsize=(8,8))
sns.countplot(x = "Outcome", data = df, palette = 'OrRd')

df.hist(figsize = (10,10), color = "Orangered")
plt.show()

#Plotting the df into histograms

plt.figure(figsize = (12,10))
sns.set_style(style = 'darkgrid')
plt.subplot(2,3,1)
sns.boxplot(x = 'Age', data = df, palette = 'OrRd')
plt.subplot(2,3,2)
sns.boxplot(x = 'BloodPressure', data = df, palette = 'OrRd')
plt.subplot(2,3,3)
sns.boxplot(x = 'BMI', data = df, palette = 'OrRd')
plt.subplot(2,3,4)
sns.boxplot(x = 'Glucose', data = df, palette = 'OrRd')
plt.subplot(2,3,5)
sns.boxplot(x = 'Insulin', data = df, palette = 'OrRd')
plt.subplot(2,3,6)
sns.boxplot(x = 'SkinThickness', data = df, palette = 'OrRd')

#Creating a box plot from the top six mean data from df.decribe
#pedigree function and pregnancies were not relevant enough so they have been ommitted for now

mean_col = ['Age', 'BloodPressure', 'BMI', 'Glucose', 'Insulin', 'SkinThickness', 'Outcome']
sns.pairplot(df[mean_col])

#creating a scatterplot to show the realtionship between the quantitative variables
#I have  also included outcome to show how each of the relevant variabels measure against the total diabetics and non-diabetics within the set

sns.boxplot(x = 'Outcome', y = 'Insulin', data = df)

#Showing the direct correlation between insulin and outcome
#As it seems so far, insulin levels are the greatest determiners of diabetes

sns.boxplot(x = 'Outcome', y = 'Age', data = df)

sns.boxplot(x = 'Outcome', y = 'Insulin', data = df)

sns.boxplot(x = 'Outcome', y = 'BMI', data = df)

plt.figure(figsize = (10,10))
sns.heatmap(df.corr())

#Plotting the correlations on a heatmap

sns.barplot(x = "BloodPressure", y = "Insulin", data = df)
plt.title("BloodPressure vs Insulin", fontsize = 12)
plt.xlabel("BloodPressure")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")

#Since there are too many values to see, in the next barplot I have chosen the data between 290 to 300 for plotting

sns.barplot(x = "BloodPressure", y = "Insulin", data = df[290:300])
plt.title("BloodPressure vs Insulin", fontsize = 12)
plt.xlabel("BloodPressure")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")

sns.barplot(x = "Glucose", y = "Insulin", data = df[290:300])
plt.title("Glucose vs Insulin", fontsize = 12)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")

#I have also chosen to compare glucose vs. insulin as it was a big factor in determining the outcome

#Train and Test data split

x = df.drop(columns = 'Outcome')

#Finding the predicted value, for the test and train I am splitting the data in a 70:30 ratio

y = df['Outcome']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

#Creating the logistic regression models based on the test and train split

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)

#Creating a confusion matrix to see the true positive and true negatives

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error

y_pred = reg.predict(x_test)
print("\nClassification Report is:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix is:\n", confusion_matrix(y_test, y_pred))
print("\nTraining Score is:\n", reg.score(x_train, y_train)*100)
print("\nR2 Score is:\n", r2_score(y_test, y_pred))
print("\nMean Squared Error is:\n", mean_squared_error(y_test, y_pred))

#Determining the accuracy score of LogReg

print("\nThe Accuracy Score is:\n", accuracy_score(y_test, y_pred)*100)

#Creating a NB model to determine accuracy

from sklearn.naive_bayes import GaussianNB

df = GaussianNB()
df.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error

y_pred = df.predict(x_test)
print("\nClassification Report is:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix is:\n", confusion_matrix(y_test, y_pred))
print("\nTraining Score is:\n", reg.score(x_train, y_train)*100)
print("\nR2 Score is:\n", r2_score(y_test, y_pred))
print("\nMean Squared Error is:\n", mean_squared_error(y_test, y_pred))

#Determining the accuracy score of NB

print("\nThe Accuracy Score is:\n", accuracy_score(y_test, y_pred)*100)

#Creating a Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier
df = RandomForestClassifier()
df.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error

y_pred = df.predict(x_test)
print("\nClassification Report is:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix is:\n", confusion_matrix(y_test, y_pred))
print("\nTraining Score is:\n", reg.score(x_train, y_train)*100)
print("\nR2 Score is:\n", r2_score(y_test, y_pred))
print("\nMean Squared Error is:\n", mean_squared_error(y_test, y_pred))

#Determining the accuracy score of NB

print("\nThe Accuracy Score is:\n", accuracy_score(y_test, y_pred)*100)

#For the final code I plan on making a visual decision tree, however, it is taking me a lot longer than expected. 
#So that will be included in the final part of the project
