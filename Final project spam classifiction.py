#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[29]:


df=pd.read_csv('C:\\Users\\HARSH ANAND\\Downloads\\mail_data.csv')


# In[ ]:





# In[30]:


import os


# In[31]:


os.getcwd()


# In[32]:


os.chdir('C:\\Users\\HARSH ANAND\\Downloads')


# In[ ]:





# In[33]:


df=pd.read_csv('C:\\Users\\HARSH ANAND\\Downloads\\mail_data.csv')


# In[34]:


print(df)


# In[35]:


data = df.where((pd.notnull(df)),'')


# In[36]:


data.head()


# In[37]:


data.info()


# In[38]:


data.shape


# In[39]:


data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1


# In[40]:


X=data['Message']

Y=data['Category']


# In[41]:


print(X)


# In[42]:


print(Y)


# In[43]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)


# In[44]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[45]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[46]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[47]:


print(X_train)


# In[48]:


print(X_train_features)


# In[49]:


model=LogisticRegression()


# In[50]:


model.fit(X_train_features,Y_train)


# In[51]:


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


# In[52]:


print('Acc on training data :', accuracy_on_training_data)


# In[53]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[54]:


print('acc on test data:',accuracy_on_test_data)


# In[55]:


input_your_mail=["Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"]
input_data_features=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)

print(prediction)
if(prediction[0]==1):
    print('ham mail')
else:
    print('spam mail')
    


# In[ ]:




