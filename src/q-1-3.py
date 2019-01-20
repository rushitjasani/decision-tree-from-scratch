#!/usr/bin/env python
# coding: utf-8

# ### q-1-3.py
# 
# Contrast the effectiveness of Misclassification rate, Gini, Entropy as impurity measures in terms of precision, recall and accuracy.

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
X = df.drop(['left'],axis='columns')


# doing One hot Encoding to convert multi-valued features to binary.

# In[4]:


Z = pd.concat([X,pd.get_dummies(X['sales'], prefix='sales')],axis='columns')
Z = pd.concat([Z,pd.get_dummies(Z['salary'], prefix='salary')],axis='columns')
Z = Z.drop(['sales','salary'],axis='columns')


# Split train and test data 

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(Z, Y,test_size=0.2)
df_ent = pd.concat([X_train,Y_train],axis=1)
df_gini = pd.concat([X_train,Y_train],axis=1)
df_mis_rate = pd.concat([X_train,Y_train],axis=1)


# mymodel function takes two argumetns : 
# 1 : dataframe
# 2 : flag
# ```
#     flag == 1 : use entropy  as impurity measure
#     flag == 2 : use gini index as impurity measure
#     flag == 3 : use misclassification rate as impurity measure
# ```

# In[6]:


def mymodel(df1, flag ):
    def get_impurity( df ):
        Class = df.keys()[-1]
        if flag == 1:
            entropy = 0
            values = df[Class].unique()
            for value in values:
                fraction = float(df[Class].value_counts()[value])/len(df[Class])
                entropy += -fraction*np.log2(fraction+eps)
            return entropy
        if flag == 2:
            entropy = 1
            values = df[Class].unique()
            for value in values:
                fraction = float(df[Class].value_counts()[value])/len(df[Class])
                entropy *= fraction
            return 2*entropy
        if flag == 3:
            entropy = 1
            values = df[Class].unique()
            for value in values:
                fraction = float(df[Class].value_counts()[value])/len(df[Class])
                entropy = min(fraction , 1-fraction)
            return entropy

    def get_impurity_attr( df,  attribute ):
        if flag == 1:
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
        if flag == 2:
            Class = df.keys()[-1]
            target_variables = df[Class].unique()  
            variables = df[attribute].unique()
            entropy2 = 0
            for variable in variables:
                entropy = 1
                for target_variable in target_variables:
                    num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
                    den = len(df[attribute][df[attribute]==variable])
                    fraction = num/(den+eps)
                    entropy *= fraction
                entropy *= 2
                fraction2 = float(den)/len(df)
                entropy2 += fraction2*entropy
            return entropy2
        if flag == 3:
            Class = df.keys()[-1]
            target_variables = df[Class].unique()  
            variables = df[attribute].unique()
            entropy2 = 0
            for variable in variables:
                entropy = 1
                for target_variable in target_variables:
                    num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
                    den = len(df[attribute][df[attribute]==variable])
                    fraction = num/(den+eps)
                    entropy = min(fraction, 1 - fraction)
                fraction2 = float(den)/len(df)
                entropy2 += fraction2*entropy
            return entropy2

    def get_subtable(df, node, value):
        return df[df[node] == value].reset_index(drop=True)

    
    def find_IG( df, val, attr  ):
        class_ent = get_impurity(df)
        left = df[df[attr] < val ].reset_index(drop=True)
        right = df[df[attr] >= val ].reset_index(drop=True)
        left_imp = get_impurity(left)
        right_imp = get_impurity(right)
        return class_ent - ((float(len(left))/(len(df)+eps) * left_imp)+( float(len(right))/(len(df)+eps) * right_imp))

    num_attrs = [ 'number_project','last_evaluation', 'satisfaction_level','average_montly_hours','time_spend_company']
    split_dict = {}
    
    def split_numerical( attr , Y , attr_name ):
        max_ig = 0
        max_split = None
        pair = pd.concat([attr, Y], axis='columns')
        pair = pair.sort_values(by =attr_name).reset_index()
        myset = set()
        for i in xrange( len(attr)-1):
            if pair['left'][i] != pair['left'][i+1] and (float(pair[attr_name][i] + pair[attr_name][i+1])/2) not in myset:
                myset.add(float(pair[attr_name][i] + pair[attr_name][i+1])/2)
                cur_ig = find_IG( pair, float(pair[attr_name][i] + pair[attr_name][i+1])/2 , attr_name )
                if cur_ig > max_ig:
                    max_ig = cur_ig
                    max_split =  float(pair[attr_name][i] + pair[attr_name][i+1])/2
        return max_split

    def attr_to_select(df):
        num_attr = [ i for i in df.columns if i in num_attrs]
        for attr in num_attr:
            split_val = split_numerical(df[attr], df['left'],attr)
            split_dict[attr] = split_val
        IG = []
        for key in df.keys()[:-1]:
            IG.append(get_impurity(df)-get_impurity_attr(df,key))
        return df.keys()[:-1][np.argmax(IG)]


    class Node:
        def __init__(self, feature):
            self.feature = feature
            self.split_val = 0
            self.positive = 0
            self.negative = 0
            self.left = None
            self.right = None

    def build_Tree(df):
        if len(df.columns) == 1:
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
            root.left = build_Tree(subtable0)
        if len(counts1)>1:
            root.right = build_Tree(subtable1)

        return root

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

    root = build_Tree(df1)
    pd.options.mode.chained_assignment = None
    X_test_copy = X_test.copy(deep=True)
    Yp=[]
    predict(X_test_copy,root,Yp)

    print confusion_matrix(Y_test,Yp)
    print classification_report(Y_test,Yp)
    print accuracy_score(Y_test, Yp)


# In[7]:


mymodel(df_ent, 1)


# In[8]:


mymodel(df_gini, 2)


# In[9]:


mymodel(df_mis_rate, 3)


# ### Observations
# 
# 1. Impurity measures impacts performace a lot
# 2. Gini and entropy performed same for most cases
# 3. Gini is easier to compute than entropy
# 4. misclassfication rate is worst performer among the three is computation is way simpler.

# In[ ]:




