# Classification-Compare

 A simple Node web app that compares accuracy of various classification algorithms on the given dataset.The user can upload a .csv file and the output is a table showing accuracy and graphs of various classification algorithms.

## This app inputs:

 A .csv file containing any kind of data for which decision has to be made on the type of classification algorithm to be selected.
 The inputs (X) should contain 2 columns and a vector (y) containing 0's and 1's.

To achieve it, it uses 7 different [Classification Algorithms](http://dataaspirant.com/2016/09/24/classification-clustering-alogrithms/) 

These 7 algorithms are :


1. [LogisticRegression](https://en.wikipedia.org/wiki/Logistic_regression)
2. [KNeighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
3. [SVC(linear)](https://en.wikipedia.org/wiki/Support_vector_machine)
4. [SVC(rbf)](https://en.wikipedia.org/wiki/Support_vector_machine)
5. [Gaussian](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
6. [Decision Tree](http://www.saedsayad.com/decision_tree.htm) 
7. [Random Forest](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)


## PREREQUISITES

```
Node Js
NPM
Python

```

## To run

```
Clone this repo
Change the path to Python as per your machine in the options variable in app.js
Cd into this repo
Npm install
Node app.js
```
Open Your favourite browser and go to localhost:3000 to access this site.
