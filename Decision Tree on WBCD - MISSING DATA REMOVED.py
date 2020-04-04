#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[2]:


data_set = pd.read_csv("breast-cancer-wisconsin.data_MISSING_DATA_ROWS_REMOVED.csv")


# In[3]:


data_clean = data_set.dropna()


# In[4]:


print(data_clean.describe())


# In[5]:


predictors = data_clean[[
 "code",
"clump_thickness",
"uniformity_of_cell_size",
"uniformity_of_cell_shape",
"marginal_adhesion",
"single_epithelial_cell_size",
"bare_nuclei",
"bland_chromatin",
"normal_nucleoli",
"mitoses"
]]


# In[6]:


targets = data_clean.target


# In[7]:


pred_train, pred_test, targ_train, targ_test = train_test_split(predictors, targets, test_size = 0.35)


# In[8]:


classifier = DecisionTreeClassifier()
classifier = classifier.fit(pred_train, targ_train)


# In[9]:


predictions = classifier.predict(pred_test)


# In[10]:


print(confusion_matrix(targ_test, predictions))
print(accuracy_score(targ_test, predictions))

