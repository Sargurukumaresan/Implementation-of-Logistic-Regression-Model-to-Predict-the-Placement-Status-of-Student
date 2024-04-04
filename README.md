# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SARGURU.K
RegisterNumber: 212222230134
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/10f0900a-503c-493a-b705-c2bddbea9cba)

## Salary Data:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/ea4b7736-61d5-4af9-9b20-5c688f868e07)

## Data-status:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/fae78309-2d1e-40cb-b35c-1b51bf774af2)

## Data duplicate:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/34f44534-b4a6-48e3-89ac-7f7a1013ef56)

## Print Data:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/b10d0e02-9b37-4d8a-800c-4c4b23dc151b)

## Checking the null function():
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/bbdf55a7-a33d-49fb-aa9d-902b68577ccb)

## Y_Prediction Array:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/c8a72db9-e82b-45c1-bb8f-433e9979c439)

## Accuracy Value:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/4c9fdb47-445f-49cc-9c99-77aece65301b)

## Confusion Value:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/34fd1ea2-1a5d-445c-ac9c-f8017b28458c)

## Classification Report:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/5031c4c0-6ca8-4e46-93da-1daf847a8d56)

## Prediction of LR:
![image](https://github.com/Abinavsankar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119103734/53c05d9e-2baf-4556-be6d-aa723cfa5838)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
