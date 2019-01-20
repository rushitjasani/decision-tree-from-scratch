#!/usr/bin/env python
# coding: utf-8

# ### q-1-1.py
# 
# Train decision tree only on categorical data. Report precision, recall, f1 score and accuracy

# STEPS : 
# 
# 1. read data
# 2. apply onehot encoding ( to handle categorical Data )
# 3. divide dataset in 80:20 for training and validation.
# 4. build decision tree using build_tree and helper functions for entropy and attribite_to_select etc
# 5. apply predict method to predict class label and use inbuilt functions to calculate confusion matrix, classification report and accuracy score.
# 6. calculate the same measures using inbuilt scikit-learn decision tree to compare performance.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
eps = np.finfo(float).eps


# Reading data from csv

# In[2]:


df = pd.read_csv('../input_data/train.csv')


# spliting features and class label and dropping numerical data.

# In[3]:


Y = df.left
X = df.drop([ 'number_project', 'left','last_evaluation', 'satisfaction_level','average_montly_hours','time_spend_company'],axis='columns')


# doing One hot Encoding to convert multi-valued features to binary.

# In[4]:


Z = pd.concat([X,pd.get_dummies(X['sales'], prefix='sales')],axis='columns')
Z = pd.concat([Z,pd.get_dummies(Z['salary'], prefix='salary')],axis='columns')
Z = Z.drop(['sales','salary'],axis='columns')


# Split data in train and test

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(Z, Y,test_size=0.2)
df1 = pd.concat([X_train,Y_train],axis=1)


# calculate entropy of class label

# In[6]:


def get_entropy( df ):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = float(df[Class].value_counts()[value])/len(df[Class])
        entropy += -fraction*np.log2(fraction+eps)
    return entropy


# calculate entropy of specific attribute

# In[7]:


def get_entropy_attr( df,  attribute ):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()  
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = float(den)/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2)


# calculate IG of all feature and select attribute with maximum IG.

# In[8]:


def attr_to_select(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        IG.append(get_entropy(df)-get_entropy_attr(df,key))
    return df.keys()[:-1][np.argmax(IG)]


# In[9]:


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


# Node structure of Decision Tree ( Binary Tree) 

# In[10]:


class Node:
    def __init__(self, feature):
        self.feature = feature
        self.positive = 0
        self.negative = 0
        self.left = None
        self.right = None


# Function to Build Tree 

# In[11]:


def build_Tree(df):
    if len(df.columns) == 1:
        return None
    node_to_split = attr_to_select(df)
    
    root = Node(node_to_split)
    root.positive = len( df[df['left']==1]['left'] )
    root.negative = len( df[df['left']==0]['left'] )
    
    subtable0 = get_subtable(df,node_to_split,0)
    subtable1 = get_subtable(df,node_to_split,1)
    
    subtable0 = subtable0.drop(node_to_split,axis=1)
    subtable1 = subtable1.drop(node_to_split,axis=1)
    
    clValue0,counts0 = np.unique(subtable0['left'],return_counts=True)
    clValue1,counts1 = np.unique(subtable1['left'],return_counts=True)
        
    if len(counts0)>1:
        root.left = build_Tree(subtable0)
    if len(counts1)>1:
        root.right = build_Tree(subtable1)
    
    return root


# In[12]:


root = build_Tree(df1)


# function to predict class label

# In[13]:


def rec_predict(df,root,Y1):
    if root == None:
        return None
    try:
        if root.right==None and root.left==None:
            Y1.append(1 if root.positive > root.negative else 0)
            return

        if root.right==None and df[root.feature] == 1:
            Y1.append(1 if root.positive > root.negative else 0)
            return 
        if root.left == None and df[root.feature] == 0:
            Y1.append(1 if root.positive > root.negative else 0)
            return
        
        if df[root.feature]==0:
            rec_predict(df,root.left,Y1)
        else:
            rec_predict(df,root.right,Y1)
    except KeyError:
        if root.left == None:
            Y1.append(1 if root.positive > root.negative else 0)
            return
        rec_predict(df,root.left,Y1)
        
def predict(df,root,Y1):
    for col,row in df.iterrows():
        rec_predict(row,root,Y1)


# My Model

# In[14]:


Y1=[]
predict(X_test,root,Y1)
print confusion_matrix(Y_test,Y1)
print classification_report(Y_test,Y1)
print accuracy_score(Y_test, Y1)


# Inbuilt scikit learn model

# In[15]:


model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

print confusion_matrix(Y_test,Y_predict)
print classification_report(Y_test,Y_predict)
print accuracy_score(Y_test, Y_predict)


# Testing from sample_test file

# In[16]:


test_df = pd.read_csv('../input_data/sample_test.csv')

Z_test = pd.concat([test_df,pd.get_dummies(test_df['sales'], prefix='sales')],axis='columns')
Z_test = pd.concat([Z_test,pd.get_dummies(Z_test['salary'], prefix='salary')],axis='columns')
Z_test = Z_test.drop(['sales','salary'],axis='columns')

Out = []
predict(Z_test, root, Out)
print Out


# ### Observations
# 
# 1. to handle categorical data in Binary decision tree, they have to be encoded using one-hot encoding or other methods. 
# 2. using only some features of the data the result is poor. 

# In[ ]:




