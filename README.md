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

Developed By: Thirugnanamoorthi.G
Register No: 212221230117

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
![o1](https://user-images.githubusercontent.com/94980741/169943529-e3fb5250-1df2-4b54-9c66-e60fc6a09fcb.png)

## Cleaning the dataset:
![o2](https://user-images.githubusercontent.com/94980741/169943569-269720ec-48f6-4ab2-a7de-778924bad19d.png)

![o3](https://user-images.githubusercontent.com/94980741/169946388-79db1eba-6d28-4ea9-84c5-f04d284b2ed7.png)


![o4](https://user-images.githubusercontent.com/94980741/169943637-1089b622-10cd-491d-a5ee-65958718ffbb.png)


## FUNCTION TRANSFORMATION:
![o6](https://user-images.githubusercontent.com/94980741/169943663-80a3ddba-2dfb-4c4a-8c4d-a7a81e5f23b7.png)
![o7](https://user-images.githubusercontent.com/94980741/169943718-1062ea1d-86ab-439c-adfa-2be8f898404b.png)


## POWER TRANSFORMATION:
![o8](https://user-images.githubusercontent.com/94980741/169943745-67bd1631-1782-4aea-855c-b97d2977041d.png)
![o9](![o10](https://user-images.githubusercontent.com/94980741/169943797-07495a19-e4e2-4af4-94bc-aaa9d0402909.png)
![o10](https://user-images.githubusercontent.com/94980741/169943831-e4344e9f-fb4b-4a3b-a8e4-306ea94d0512.png)
![o11](https://user-images.githubusercontent.com/94980741/169946410-839fa8c7-3b6a-4162-878e-9fad6471e4e7.png)

![o12](https://user-images.githubusercontent.com/94980741/169943924-c9de2460-a96d-4432-a663-c61349b22855.png)



## QUANTILE TRANSFORMATION
![o13](https://user-images.githubusercontent.com/94980741/169943947-aec46d8c-f24b-42de-982b-cec539bc288c.png)
![o14](https://user-images.githubusercontent.com/94980741/169943959-14e6ff3a-1255-4618-b838-f1556371cf4a.png)
![o15](https://user-images.githubusercontent.com/94980741/169945593-ba24672f-e94c-42a0-bce9-f066da9b840b.png)
![o16](https://user-images.githubusercontent.com/94980741/169945611-b9b671f5-2995-4879-94bc-c8154dd4ea0e.png)

## Final Result:
![o17](https://user-images.githubusercontent.com/94980741/169945629-8006f068-d26a-4464-b8c2-923432e7ccd1.png)
![o19](https://user-images.githubusercontent.com/94980741/169945685-3830ad99-ec9a-4e70-b76a-30a55bce21e9.png)



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
![s1](https://user-images.githubusercontent.com/94980741/169945732-a3c5e53c-af67-433f-a832-53c594d419a6.png)
![s2](https://user-images.githubusercontent.com/94980741/169945793-5fe88a2c-34c7-493c-95cc-189fdd136b55.png)


## FUNCTION TRANSFORMATION:
![s3](https://user-images.githubusercontent.com/94980741/169945811-b6ce7e69-715e-4260-a111-574afe30c3dc.png)
 ![s4](https://user-images.githubusercontent.com/94980741/169945862-3edd294e-3a4d-4ce4-826d-0baa887ce665.png)
![s5](https://user-images.githubusercontent.com/94980741/169945876-d8c4e20e-7c10-45c0-85e7-62cf90bd3e36.png)
![s6](https://user-images.githubusercontent.com/94980741/169945908-ea310faf-7bb6-44ed-95e5-7a2e14d286da.png)


## POWER TRANSFORMATION:
![s7](https://user-images.githubusercontent.com/94980741/169945976-98bfb49c-2eb8-4b01-9276-1188cbd9236d.png)
![s8](https://user-images.githubusercontent.com/94980741/169945989-03ca665f-9901-4383-b2e5-854ecddadbe3.png)
![s9](https://user-images.githubusercontent.com/94980741/169946000-427861de-ff8c-4ac7-95ed-5e886f031087.png)
![s10](https://user-images.githubusercontent.com/94980741/169946021-560ab6e0-1219-4cdd-9474-191653a26b69.png)

## QUANTILE TRANSFORAMATION:
![s12](https://user-images.githubusercontent.com/94980741/169946056-2ba19c74-9377-4f6a-9f3e-abefb44b39d0.png)
![s13](https://user-images.githubusercontent.com/94980741/169946065-b4c84953-73a0-46bf-8bcf-2e98c636c6e4.png)
![s14](https://user-images.githubusercontent.com/94980741/169946073-a6c4d48e-cf08-48e1-b571-3d026f802fe3.png)
![s15](https://user-images.githubusercontent.com/94980741/169946087-1794afbc-77bd-4ee7-a5b6-ee7cab392dc0.png)
![s17](https://user-images.githubusercontent.com/94980741/169946103-a7bf2c78-6100-4a0c-a610-fa60fb77bea0.png)
![s18](https://user-images.githubusercontent.com/94980741/169946130-65c274f8-427a-41b8-946a-6a29d9d21076.png)
![s19](https://user-images.githubusercontent.com/94980741/169946140-c33f75c1-a344-42c5-abb3-d9c019081904.png)

## Final Result:
![s20](https://user-images.githubusercontent.com/94980741/169946158-7c7f50dd-a0e5-4f39-8b90-c32299a54dba.png)
![s21](https://user-images.githubusercontent.com/94980741/169946166-7afb3420-38fb-4e6d-8c87-c3fdf070877f.png)



# Result:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
