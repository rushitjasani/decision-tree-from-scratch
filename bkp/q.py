import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
eps = np.finfo(float).eps

from pylab import *
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

df1 = df[ df['left'] == 0]
df2 = df[ df['left'] == 1]

figure()

scatter(df1['satisfaction_level'],df1['last_evaluation'],marker='+' )
scatter(df2['satisfaction_level'],df2['last_evaluation'],marker='+')

xlabel('satisfaction_level')
ylabel('last_evaluation')
title('scatter plot')

show()
