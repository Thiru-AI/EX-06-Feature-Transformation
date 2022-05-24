# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
'''
Developed By: Balaji.K
Register No: 212221230011
'''
```
# titanic_dataset.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  
#ReciprocalTransformation  
np.reciprocal(df["Age"])  
#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  
df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  
df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  
df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  

#QUANTILE TRANSFORMATION  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  
df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  
sm.qqplot(df['Age_1'],line='45')  
plt.show()  
df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  
sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df 
```
# OUTPUT
## Reading the data set
![output](./titanic_dataset/o1.png)
## Cleaning the dataset:
![output](./titanic_dataset/o2.png)
![output](./titanic_dataset/o3.png)
![output](./titanic_dataset/o4.png)
## FUNCTION TRANSFORMATION:
![output](./titanic_dataset/o6.png)
![output](./titanic_dataset/o7.png)
## POWER TRANSFORMATION:
![output](./titanic_dataset/o8.png)
![output](./titanic_dataset/o9.png)
![output](./titanic_dataset/o10.png)
![output](./titanic_dataset/o11.png)
![output](./titanic_dataset/o12.png)
## QUANTILE TRANSFORMATION
![output](./titanic_dataset/o13.png)
![output](./titanic_dataset/o14.png)
![output](./titanic_dataset/o15.png)
![output](./titanic_dataset/o16.png)
## Final Result:
![output](./titanic_dataset/o17.png)
![output](./titanic_dataset/o19.png)

# data_to_transform.csv
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  
df=pd.read_csv("Data_To_Transform.csv")  
df  
df.skew()  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Highly Positive Skew"])  
#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])  
#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])  
#Square Transformation  
np.square(df["Highly Negative Skew"])  

#POWER TRANSFORMATION:  
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df  
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df  
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df  
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df  

#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')  
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()  
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show()  

df.skew()  
df 
```

# Output:
## Reading the data set:
![output](./Transform_dataset/s1.png)
![output](./Transform_dataset/s2.png)
## FUNCTION TRANSFORMATION:
![output](./Transform_dataset/s3.png)
![output](./Transform_dataset/s4.png)
![output](./Transform_dataset/s5.png)
![output](./Transform_dataset/s6.png)
## POWER TRANSFORMATION:
![output](./Transform_dataset/s7.png)
![output](./Transform_dataset/s8.png)
![output](./Transform_dataset/s9.png)
![output](./Transform_dataset/s10.png)
## QUANTILE TRANSFORAMATION:
![output](./Transform_dataset/s12.png)
![output](./Transform_dataset/s13.png)
![output](./Transform_dataset/s14.png)
![output](./Transform_dataset/s15.png)
![output](./Transform_dataset/s17.png)
![output](./Transform_dataset/s18.png)
![output](./Transform_dataset/s19.png)
## Final Result:
![output](./Transform_dataset/s20.png)
![output](./Transform_dataset/s21.png)


# Result:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
