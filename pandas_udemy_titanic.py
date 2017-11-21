# to handle datasets
import pandas as pd
import numpy as np
# for text / string processing
import re
# for plotting
import matplotlib.pyplot as plt
 # to divide train and test set
from sklearn.model_selection import train_test_split
## feature scaling
from sklearn.preprocessing import MinMaxScaler
## for tree binarisation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
##to build the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
##to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
pd.pandas.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

##load dataset
data = pd.read_csv('train.csv')
submission = pd.read_csv('test.csv')

##Graph Age and Fare to see distribution. Age is Gausian while Fare is not. 
##We'll use Gaussian for Age and interquartile range for Fare in processing
#plt.figure(figsize=(15,6))
#plt.subplot(1, 2, 1) 
#fig = data.Age.hist(bins=20)
#fig.set_ylabel('Number of passengers')
#fig.set_xlabel('Age')
 
#plt.subplot(1, 2, 2)
#fig = data.Fare.hist(bins=20)
#fig.set_ylabel('Number of passengers')
#fig.set_xlabel('Fare')

##Age Gaussian
Upper_boundary = data.Age.mean() + 3* data.Age.std()
Lower_boundary = data.Age.mean() - 3* data.Age.std()
#print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
##Fare IQR
IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
Lower_fence = data.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = data.Fare.quantile(0.75) + (IQR * 3)
#print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

##Find outliers in categorical variables. Anything representing less than 1% of data is considered an outlier
##Values greater than 4 in SibSp are rare. Values bigger than 2 are rare in Parch
for var in ['Pclass',  'SibSp', 'Parch']:
    ##counts the number of times each value occurrs and then divides it by the length of data which is 891
    print(data[var].value_counts() / np.float(len(data)))
 





















































