#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt


# In[3]:


iris = load_iris()


# In[4]:


print(iris.feature_names)


# In[5]:


print(iris.target_names)


# In[7]:


print(iris.target[0])


# In[9]:


for i in range(len(iris.target)):
    print("Ex", i, ": label", iris.target[i], "features", iris.data[i])


# In[10]:


print("Ex 0 : label",iris.target[0], "features", iris.data[0] )
print("Ex 50 : label",iris.target[50], "features", iris.data[50] )
print("Ex 100 : label",iris.target[100], "features", iris.data[100] )


# In[12]:


test_idx = [0, 50, 100]


# In[13]:


#training
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)


# In[14]:


#t testing
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# In[18]:


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


# In[19]:


print(test_target)
print(clf.predict(test_data))


# In[20]:


print(clf.predict([[7.9, 3.1, 5.1, 1.0]]))
print(clf.predict([[5.9, 4.2, 2.5, 1.4]]))
print(clf.predict([[4.4, 4.2, 3.5, 1.2]]))


# In[ ]:




