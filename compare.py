# -*- coding: utf-8 -*-

# Classification template

# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
# Importing the dataset
dataset = pd.read_csv(sys.argv[1])
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test = sc.transform(X_test.astype(float))

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



classifiers = { 'LogisticRegression':LogisticRegression(random_state = 0),
              'KNeighbors': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
               'SVC(linear)': SVC(kernel = 'linear', random_state = 0),
               'SVC(rbf)': SVC(kernel = 'rbf', random_state = 0),
              'Gaussian': GaussianNB(),
               'Decision Tree':DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
               'Random Forest':RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
             }
n_classifiers = len(classifiers)
#abc ={'name':'k','accuracy':'k','plot':'k'}
data = [ ]
#for x in range(n_classifiers):
#    data.append(abc)
i=0
a = np.array(y_test)

b = a.ravel()

for index, (name, classifier) in enumerate(classifiers.items()):
     classifier.fit(X_train, y_train)
     y_pred = classifier.predict(X_test)
   #  accuracy_score=accuracy_score(np.transpose(y_test),np.transpose( y_pred))
#     data[i]['name']=name
#     data[i]['accuracy']=8
#     data[i]['plot']='./plotPics/'+name+'.png'
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred)
     er=cm[0][1]+cm[1][0]
     co=cm[1][1]+cm[0][0]
     tot=er+co
     obj={
             'name':name,
             'accuracy':(float(co)/float(tot)),
             'plot':'./plotPics/'+name+'.png'
             
        }
     data.append(obj)
 # Visualising the Test set results
     from matplotlib.colors import ListedColormap
     X_set, y_set = X_test, y_test
     X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
     plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
     plt.xlim(X1.min(), X1.max())
     plt.ylim(X2.min(), X2.max())
    
     for i, j in enumerate(np.unique(y_set)):
         plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                     c = ListedColormap(('red', 'green'))(i), label = j)
        
     plt.title('Classifier (Test set) ' )
     plt.xlabel('Age')
     plt.ylabel('Estimated Salary')
     plt.legend()
     #plt.show()
     plt.savefig('./plotPics/'+name+'.png')
     plt.clf() 
#data = json.loads(data)
#json_string = json.dumps(data)
with open('output', 'w') as outfile:
    json.dump(data, outfile)
print(data)
sys.stdout.flush()    