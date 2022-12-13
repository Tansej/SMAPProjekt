import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

#Path to dataset
pathDataset = 'C:\\Users\\Tansej\\Desktop\\SMAPProjekt\\Projekt\\dataset\\mails.csv'
print('Dataset loading from...' + pathDataset)

#Read mails and sort them to x=message and y=status
mails = pd.read_csv(pathDataset)
print('Mails from csv loaded.')
x = mails['message']
y = mails['status']

#Split training and test data, test data = 25% of total data
print('Splitting training and testing data.')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
print('Split complete!')
