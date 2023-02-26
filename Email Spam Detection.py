#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


# In[5]:


email = pd.read_csv("Desktop/spam.csv",encoding ='ISO-8859-1')
email


# In[6]:


email = email.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
email = email.dropna()
email.info()


# In[7]:


X = email['v2'].values
y = email['v1'].values


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


# In[11]:


from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state=0)
svm.fit(X_train,y_train)


# In[12]:


y_pred = svm.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy*100)


# In[ ]:




