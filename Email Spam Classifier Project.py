#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('mail_data.csv')


# In[3]:


print(df)


# In[4]:


data = df.where((pd.notnull(df)), '')


# In[5]:


data.head()


# In[6]:


data.info()


# In[63]:


data.shape


# In[8]:


data.loc[data['Category']== 'spam','Category',]=0
data.loc[data['Category']== 'ham','Category',]=1


# In[9]:


X=data['Message']
Y=data['Category']


# In[10]:


print(X)


# In[11]:


print(Y)


# In[77]:


# Example: Class distribution visualization
import seaborn as sns
sns.countplot(x='Category', data=data)
plt.title('Class Distribution - Spam vs. Ham')
plt.show()


# In[78]:


# Example: Analyzing message length
data['Message Length'] = data['Message'].apply(len)
sns.histplot(data, x='Message Length', hue='Category', bins=50, kde=True)
plt.title('Distribution of Message Lengths - Spam vs. Ham')
plt.show()


# In[12]:


X_train, X_test ,Y_train,Y_test =train_test_split(X,Y, test_size=0.2, random_state =3)


# In[13]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[14]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[52]:


feature_extraction= TfidfVectorizer(min_df =1, stop_words = 'english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[53]:


print(X_train)


# In[54]:


print(X_train_features)


# In[55]:


model = LogisticRegression()


# In[56]:


model.fit(X_train_features, Y_train)


# In[57]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data= accuracy_score(Y_train,prediction_on_training_data)


# In[58]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[59]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)


# In[60]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[62]:


input_your_mail= ["This is the 2nd time we have tried to contact you.u have won $400 price.2 claim is easy just call +123 456 789"]

input_data_features = feature_extraction.transform(input_your_mail)


prediction = model.predict(input_data_features)
 
print(prediction)

if(prediction[0]==1):
 print('Ham mail')
else:
 print('spam mail')


# In[ ]:





# In[ ]:





# In[ ]:




