#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Loan_Default.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.drop(df[df['Gender'] == 'Sex Not Available'].index, inplace=True)


# In[7]:


df.head()


# In[8]:


df.isna().sum()


# In[9]:


df.fillna({
    'income': df.income.median(),
    'rate_of_interest': df['rate_of_interest'].mode()[0],
    'Interest_rate_spread' : df['Interest_rate_spread'].mode()[0],
    'Upfront_charges': df['Upfront_charges'].mode()[0]
}, inplace=True)


# In[10]:


df.Status.value_counts()


# In[11]:


df['rate_of_interest'].value_counts()


# In[12]:


df.dropna(inplace=True)


# In[13]:


df.Status.value_counts()


# In[14]:


df.isna().sum()


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


df.dtypes


# In[18]:


df.head()


# In[20]:


df.reset_index(inplace=True)


# In[21]:


df.head()


# In[22]:


df.drop(["index"], axis=1, inplace=True)


# In[23]:


df.columns


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


le = LabelEncoder()


# In[26]:


X = df.drop("Status", axis=1)
y = df.Status


# In[27]:


X.head()


# In[28]:


X.columns


# In[29]:


X.drop(['ID', "year"], axis=1, inplace=True)


# In[30]:


X.head()


# In[31]:


for col in list(X.columns):
    X[col] = le.fit_transform(X[col])


# In[32]:


X.head()


# In[33]:


y.value_counts()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)


# In[38]:


X_train.shape


# In[39]:


y_train.shape


# # Random Forest Classifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier


# In[41]:


rfs = RandomForestClassifier(n_estimators = 50, max_depth = 5)


# In[42]:


rfs.fit(X_train, y_train)


# In[43]:


train_pred = rfs.predict(X_train)


# In[44]:


n = np.arange(len(X_train))
plt.scatter(n, y_train, color='red', label='actual')
plt.scatter(n, train_pred, color='blue', label='predicted')
plt.legend()
plt.show()


# In[48]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


# In[49]:


accuracy_score(y_train, train_pred)


# In[50]:


f1_score(y_train, train_pred)


# In[52]:


cm = confusion_matrix(y_train, train_pred)
cm


# In[59]:


sns.heatmap(cm, cmap='viridis', annot=True, fmt='.2f')
plt.show()


# In[57]:


test_pred = rfs.predict(X_test)


# In[58]:


n = np.arange(len(X_test))
plt.scatter(n, y_test, color='red', label='actual')
plt.scatter(n, test_pred, color='blue', label='predicted')
plt.legend()
plt.show()


# In[60]:


accuracy_score(y_test, test_pred)


# In[61]:


precision_score(y_test, test_pred)


# In[63]:


recall_score(y_test, test_pred)


# In[64]:


f1_score(y_test, test_pred)


# In[65]:


cm = confusion_matrix(y_test, test_pred)
cm


# In[66]:


sns.heatmap(cm, cmap='viridis', annot=True, fmt='.2f')
plt.show()


# In[67]:


print(classification_report(y_test, test_pred))

