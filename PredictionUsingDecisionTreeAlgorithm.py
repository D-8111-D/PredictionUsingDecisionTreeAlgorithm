
# Deepshikha Prajapati

# Prediction Using Decision Tree Algorithm

# Importing All Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn import tree
import dataset

data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

# Showing first 5 values

df.head()

# Showing Last 5 values

df.tail()

# checking for null values

df.isnull().sum()

# Number of rows and columns

df.shape
(150, 5)

# Showing only target data (Dependent Variable)

print(df["target"])

# Splitting Data

fc = [x for x in df.columns if x!="target"]
x= df[fc]
y= df["target"]
X_train, X_test, Y_train, Y_test = train_test_split(x,y, random_state = 100, test_size = 0.30)

# Display of Data

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Building Desicion tree model

model1 = DecisionTreeClassifier()

Y_pred = model1.predict(X_test)

data2 = pd.DataFrame({"Actual":Y_test,"Predicted":Y_pred})
data2.head()

# Testing the accuracy of model prediction

accuracy_score(Y_test,Y_pred)
0.9555555555555556

# Plotting

f_n = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
c_n = ["Setosa", "Versicolor", "Virginica"]
plot_tree(model1,feature_names = f_n, class_names = c_n , filled = True)

modelx= DecisionTreeClassifier().fit(x,y)

plt.figure(figsize = (20,15))
tree = plot_tree(modelx, feature_names = f_n, class_names = c_n, filled = True)

