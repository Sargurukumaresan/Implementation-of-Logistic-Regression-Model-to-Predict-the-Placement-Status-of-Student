# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SARGURU 
RegisterNumber: 21222223034
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
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
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
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
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement_data

![Screenshot 2024-03-15 211814](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/8b60556e-1836-4040-9641-64dec989a7e9)


### Salary_data
![Screenshot 2024-03-15 210732](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/24a76033-f13f-4447-972d-8214f814ab44)

### ISNULL()
![image](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/39949634-5b22-432c-8637-3a04527145d0)

### DUPLICATED()
![Screenshot 2024-03-15 210927](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/be4a4686-74a4-4868-b5e3-e9d26b99edbc)

### Print Data
![Screenshot 2024-03-15 211440](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/5f2d84c4-b065-4cc4-85e8-8fd388af4767)

### iloc[:,:-1]
![Screenshot 2024-03-15 211207](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/97b2ca25-5ee4-40af-8751-ef19b496a6a8)

### Data_Status
![Screenshot 2024-03-15 211231](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/d5e101c3-0631-4180-aeea-8d8dcc2c0c3d)

### Y_Prediction array:
![Screenshot 2024-03-15 211300](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/3bc1e362-4b02-418c-baaf-e8816550eca5)

### Accuray value:
![Screenshot 2024-03-15 211326](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/fc0ecbe7-41eb-40eb-bc0e-8def63527700)

### Confusion Array:
![Screenshot 2024-03-15 211519](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/f381506f-d736-4d2f-8755-1a139dde3ec6)

### Classification report:
![Screenshot 2024-03-15 211545](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/83e9c53e-2349-485e-b0f9-c814786e6931)

### Prediction of LR:
![Screenshot 2024-03-15 211613](https://github.com/DEVADARSHAN2/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119432150/b3834679-b1c1-4d5d-b10f-b97d7a8f3b64)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
