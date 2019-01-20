#!/usr/bin/env python
# coding: utf-8

# ### q-1-5.py
# 
# Plot a graph of training and validation error with respect to depth of your decision tree. Also plot the training and validation error with respect to number of nodes in the decision tree.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
eps = np.finfo(float).eps

from pylab import *
import matplotlib
import matplotlib.pyplot as plt


# Reading data from csv

# In[2]:


df = pd.read_csv('../input_data/train.csv')


# spliting features and class label and dropping numerical data.

# In[3]:


Y = df.left
X = df.drop(['left'],axis='columns')


# doing One hot Encoding to convert multi-valued features to binary.

# In[4]:


Z = pd.concat([X,pd.get_dummies(X['sales'], prefix='sales')],axis='columns')
Z = pd.concat([Z,pd.get_dummies(Z['salary'], prefix='salary')],axis='columns')
Z = Z.drop(['sales','salary'],axis='columns')


# Split train and test data 

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(Z, Y,test_size=0.2)
X_test_copy = X_test.copy(deep=True)
df1 = pd.concat([X_train,Y_train],axis=1)


# finding entropy of class label

# In[6]:


def get_entropy( df ):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = float(df[Class].value_counts()[value])/len(df[Class])
        entropy += -fraction*np.log2(fraction+eps)
    return entropy


# finding entropy of given attribute

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


# In[8]:


num_attrs = [ 'number_project','last_evaluation', 'satisfaction_level','average_montly_hours','time_spend_company'] 
split_dict = {}


# dividing table in subtable by given value and attribute.

# In[9]:


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


# finding information gain by splitting value of attr at splitting point val.

# In[10]:


def find_IG( df, val, attr ):
    class_ent = get_entropy(df)
    left = df[df[attr] < val ].reset_index(drop=True)
    right = df[df[attr] >= val ].reset_index(drop=True)
    left_ent = get_entropy(left)
    right_ent = get_entropy(right)
    return class_ent - ((float(len(left))/(len(df)+eps) * left_ent)+( float(len(right))/(len(df)+eps) * right_ent))


# finding maximum split point by trying all possibilities

# In[11]:


def split_numerical( attr , Y , attr_name):
    max_ig = 0
    max_split = None
    pair = pd.concat([attr, Y], axis='columns')
    pair = pair.sort_values(by =attr_name).reset_index()
    myset = set()
    for i in xrange( len(attr)-1):
        if pair['left'][i] != pair['left'][i+1] and  (float(pair[attr_name][i] + pair[attr_name][i+1])/2) not in myset:
            myset.add(float(pair[attr_name][i] + pair[attr_name][i+1])/2)
            cur_ig = find_IG( pair, float(pair[attr_name][i] + pair[attr_name][i+1])/2 , attr_name )
            if cur_ig > max_ig:
                max_ig = cur_ig
                max_split =  float(pair[attr_name][i] + pair[attr_name][i+1])/2
    return max_split


# selecting attribute with highest information gain

# In[12]:


def attr_to_select(df):
    num_attr = [ i for i in df.columns if i in num_attrs]
    for attr in num_attr:
        split_val = split_numerical(df[attr], df['left'],attr)
        split_dict[attr] = split_val
    IG = []
    for key in df.keys()[:-1]:
        IG.append(get_entropy(df)-get_entropy_attr(df,key))
    return df.keys()[:-1][np.argmax(IG)]


# Tree node structures

# In[13]:


class Node:
    def __init__(self, feature):
        self.feature = feature
        self.split_val = 0
        self.positive = 0
        self.negative = 0
        self.left = None
        self.right = None


# Building Tree

# In[14]:


def build_Tree(df, depth):
    if len(df.columns) == 1 or depth == 0:
        return None
    node_to_split = attr_to_select(df)
    
    root = Node(node_to_split)
    subtable0 = []
    subtable1 = []
    if node_to_split in num_attrs:
        split_point = split_dict[node_to_split]
        
        root.split_val = split_point
        root.positive = len( df[df['left'] >= split_point]['left'] )
        root.negative = len( df[df['left'] <  split_point]['left'] )
        
        subtable0 = df[df[node_to_split] < split_point].reset_index(drop=True)
        subtable1 = df[df[node_to_split] >= split_point].reset_index(drop=True)
        
    else:
        root.positive = len( df[df['left']==1]['left'] )
        root.negative = len( df[df['left']==0]['left'] )
        
        subtable0 = get_subtable(df,node_to_split,0)
        subtable1 = get_subtable(df,node_to_split,1)
    
    
    subtable0 = subtable0.drop(node_to_split,axis=1)
    subtable1 = subtable1.drop(node_to_split,axis=1)
    
    clValue0,counts0 = np.unique(subtable0['left'],return_counts=True)
    clValue1,counts1 = np.unique(subtable1['left'],return_counts=True)
        
    if len(counts0)>1:
        root.left = build_Tree(subtable0, depth-1)
    if len(counts1)>1:
        root.right = build_Tree(subtable1, depth-1)
    
    return root


# Prediction od data

# In[15]:


def rec_predict(df,root,Y1):
    if root == None:
        return None
        
    if root.feature in num_attrs:
        try:
            if root.right==None and root.left==None:
                Y1.append(1 if root.positive > root.negative else 0)
                return

            if root.right==None and df[root.feature] >= root.split_val:
                Y1.append(1 if root.positive > root.negative else 0)
                return 
            if root.left == None and df[root.feature] < root.split_val:
                Y1.append(1 if root.positive > root.negative else 0)
                return

            if df[root.feature]< root.split_val:
                rec_predict(df,root.left,Y1)
            else:
                rec_predict(df,root.right,Y1)
        except KeyError:
            if root.left == None:
                Y1.append(1 if root.positive > root.negative else 0)
                return
            rec_predict(df,root.left,Y1)
    else:
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


# In[16]:


final_ans_train = []
final_ans_test = []
for i in xrange(2,21,2):
    root = build_Tree(df1,i)
    
    Yp_train=[]
    Yp_test=[]
    
    xtr_cp = X_train.copy(deep=True)
    xts_cp = X_test.copy(deep=True)
    
    predict(xtr_cp,root,Yp_train)
    predict(xts_cp,root,Yp_test)
    
    final_ans_train.append(1 - accuracy_score(Y_train, Yp_train))
    final_ans_test.append(1 - accuracy_score(Y_test, Yp_test))


# In[17]:


x_axis = [i for i in xrange(2,21,2)]

fig, ax = plt.subplots()

ax.plot(x_axis, final_ans_train, label="training error", marker='o')
ax.plot(x_axis, final_ans_test, label="testing error", marker='o')
ax.legend(loc=1) 
ax.set_xlabel("depth")
ax.set_ylabel("error")
ax.set_title('depth vs error')
print final_ans_train
print final_ans_test


# In[18]:


count = 0
def build_Tree_node(df, depth):
    if len(df.columns) == 1 or depth == 0:
        return None
    node_to_split = attr_to_select(df)
    global count
    root = Node(node_to_split)
    count+=1
    subtable0 = []
    subtable1 = []
    if node_to_split in num_attrs:
        split_point = split_dict[node_to_split]
        
        root.split_val = split_point
        root.positive = len( df[df['left'] >= split_point]['left'] )
        root.negative = len( df[df['left'] <  split_point]['left'] )
        
        subtable0 = df[df[node_to_split] < split_point].reset_index(drop=True)
        subtable1 = df[df[node_to_split] >= split_point].reset_index(drop=True)
        
    else:
        root.positive = len( df[df['left']==1]['left'] )
        root.negative = len( df[df['left']==0]['left'] )
        
        subtable0 = get_subtable(df,node_to_split,0)
        subtable1 = get_subtable(df,node_to_split,1)
    
    
    subtable0 = subtable0.drop(node_to_split,axis=1)
    subtable1 = subtable1.drop(node_to_split,axis=1)
    
    clValue0,counts0 = np.unique(subtable0['left'],return_counts=True)
    clValue1,counts1 = np.unique(subtable1['left'],return_counts=True)
        
    if len(counts0)>1:
        root.left = build_Tree_node(subtable0, depth-1)
    if len(counts1)>1:
        root.right = build_Tree_node(subtable1, depth-1)
    
    return root


# In[ ]:





# In[19]:


final_ans_train = []
final_ans_test = []
x_axis = []
for i in xrange(2,21,2):
    global count
    count = 0
    root = build_Tree_node(df1,i)
    x_axis.append(count)
    print "depth " , i , "count ", count
    Yp_train=[]
    Yp_test=[]
    
    xtr_cp = X_train.copy(deep=True)
    xts_cp = X_test.copy(deep=True)
    
    predict(xtr_cp,root,Yp_train)
    predict(xts_cp,root,Yp_test)
    
    final_ans_train.append( 1 - accuracy_score(Y_train, Yp_train))
    final_ans_test.append(1 - accuracy_score(Y_test, Yp_test))


# In[ ]:





# In[20]:


fig, ax = plt.subplots()

ax.plot(x_axis, final_ans_train, label="training error", marker='o')
ax.plot(x_axis, final_ans_test , label="testing error", marker='o')
ax.legend(loc=1)
ax.set_xlabel("node")
ax.set_ylabel("error")
ax.set_title('number of node vs error')
print final_ans_train
print final_ans_test


# Our Model
root = build_Tree(df1)pd.options.mode.chained_assignment = None

Yp=[]
# preprocess(X_test,split_dict)
predict(X_test,root,Yp)

print confusion_matrix(Y_test,Yp)
print classification_report(Y_test,Yp)
print accuracy_score(Y_test, Yp)
# Inbuilt scikit learn model
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test_copy)
print confusion_matrix(Y_test,Y_predict)
print classification_report(Y_test,Y_predict)
print accuracy_score(Y_test, Y_predict)
# Testing on sample_test.csv
test_df = pd.read_csv('../input_data/testing.csv')
Y_act = test_df.left
test_df = test_df.drop(['left'],axis='columns')

Z_test = pd.concat([test_df,pd.get_dummies(test_df['sales'], prefix='sales')],axis='columns')
Z_test = pd.concat([Z_test,pd.get_dummies(Z_test['salary'], prefix='salary')],axis='columns')
Z_test = Z_test.drop(['sales','salary'],axis='columns')

Out = []
predict(Z_test, root, Out)
print confusion_matrix(Out,Y_act)
print classification_report(Out,Y_act)
print accuracy_score(Out,Y_act)
# ### Observation
# 
# 1. when depth or number of nodes are less, training and validation error is high and variation between them is very less.
# 2. when depth or number of nodes are more, training and validation error is low and validation error is little more than training error. 

# In[ ]:




