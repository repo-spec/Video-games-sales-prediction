#!/usr/bin/env python
# coding: utf-8

# In[25]:

#Importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[2]:

#Reading file
df=pd.read_csv('D://Machine learning Self projects/vgsales.csv')

#Exploratory Data Analysis
# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[17]:


df.duplicated().sum()


# In[3]:


df1=df.copy()


# In[4]:


df1.dropna(inplace=True)


# In[23]:


df1.isnull().sum()


# In[9]:


df1.head()


# In[23]:


len(df1)


# In[24]:


len(df)





# In[5]:


df1['Year']=pd.to_datetime(df1['Year'],format='%Y')


# In[6]:


df1.head()


# In[6]:


df1['Year']=df1['Year'].dt.year


# In[53]:


df1['Year']


# In[54]:


df1.head()


# In[55]:


df1.isna().sum()


# In[8]:


df1['Platform'].unique()


# In[7]:


arr=df1['Platform'].unique()


# In[59]:


arr


# In[9]:


dict1=dict(enumerate(arr))


# In[66]:


dict1


# In[76]:


dict1.items()


# In[77]:


dict1.keys()


# In[11]:


del df1['Platform_type']


# In[81]:


df1.head()


# In[27]:


arr1=[]


# In[22]:


df1['Platform']


# In[26]:


for key in dict1:
    print(dict1[key])


# In[10]:


arr1=[]


# In[11]:


for platform in df1['Platform']:
     for key in dict1:
            if platform==dict1[key]:
                arr1.append(key)
                        
    
df1['Platform_type']=arr1
    


# In[17]:


print(arr1)


# In[11]:


df1.head()


# In[42]:


len(df1['Publisher'].unique())


# In[43]:


len(df1['Genre'].unique())


# In[12]:


arr2=df1['Genre'].unique()
dict2=dict(enumerate(arr2))
dict2


# In[ ]:





# In[13]:


arr3=[]
for platform in df1['Genre']:
     for key in dict2:
            if platform==dict2[key]:
                arr3.append(key)
                        
    
df1['Genre_type']=arr3
    


# In[15]:


df1.head()


# In[49]:


sns.boxplot(df1['Other_Sales'])


# In[31]:

#Train and test data
train=df1.iloc[0:800]
test=df1.iloc[801:1600]


# In[32]:


x_train=train.drop(columns=['Global_Sales','Name','Genre','Platform','Publisher'],axis=1)
y_train=train['Global_Sales']


# In[33]:


x_test=test.drop(columns=['Global_Sales','Name','Genre','Platform','Publisher'],axis=1)
y_test=test['Global_Sales']


# In[31]:


df1.head()


# In[34]:

#Creating model using linear regression
model=LinearRegression()
model.fit(x_train,y_train)


# In[62]:


print(model.intercept_)


# In[63]:


print(model.coef_)


# In[35]:


pred_train=model.predict(x_train)


# In[36]:


rmse_error=mean_squared_error(y_train,pred_train)**0.5


# In[70]:


rmse_error


# In[37]:


pred_test=model.predict(x_test)
rmse_test=mean_squared_error(y_test,pred_test)**0.5
rmse_test


# In[76]:


print(x_train)


# In[79]:


y_train


# In[80]:


pred_train


# In[24]:


plt.plot(x_train,pred_train,'+',color='green')
plt.title('Performance train')
plt.xlabel('Input')
plt.ylabel('Predictions')
plt.show()


# In[41]:


print(len(x_train),len(y_train))


# In[44]:


plt.plot(x_test,pred_test,'+',color='green')
plt.title('Performance test')
plt.xlabel('Input')
plt.ylabel('Predictions')
plt.show()


# In[ ]:




