from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import timeit
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings
from sklearn.metrics import precision_score, recall_score
warnings.filterwarnings("ignore")

#Getting MNIST data
mnist = fetch_mldata('MNIST original')
#Dividing the data into Label and Target
X,y=mnist["data"],mnist["target"]

#Dividing the data into Training Dataset and Testing Dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#stochastic gradient descent Model
sgd_clf = SGDClassifier()

#Traing my model
sgd_clf.fit(X_train, y_train)

#prediction
y_pred_sgd = sgd_clf.predict(X_test)

#Accuracy
acc_sgd = accuracy_score(y_test, y_pred_sgd)

print ("stochastic gradient descent accuracy: ",acc_sgd)

"""Sometimes You will get 99% Accuracy but you need to check the cross validation values 
and precision score as well as the recall score. """

cross_val=cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("Cross Validation Value:",cross_val)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

#Confussion matrix
conf_mx=confusion_matrix(y_train, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print("Precision Score:",ps)
ps=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",ps)
"""Here i am showing you the graph of the confussion matrix. more the bright more erroneous it is. 
    so by getting the confussion matrix graph we canlearn from it and we can change the values of 
    classifier or by removing the noise from the images we can get the optimized result"""
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
########################################################################################################

#Random Forest Classifier
clf_rf = RandomForestClassifier()

#Traing my model
clf_rf.fit(X_train, y_train)

#Making the prediction
y_pred_rf = clf_rf.predict(X_test)

#Measuring the accuracy of machine
acc_rf = accuracy_score(y_test, y_pred_rf)
print ("random forest accuracy: ",acc_rf)

#Cross Validation
cross_val=cross_val_score(clf_rf, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_rf, X_train, y_train, cv=3)
print(y_train_pred)

#Confussion matrix
conf_mx=confusion_matrix(y_train, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print("Precision Score:",ps)
ps=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",ps)
"""Here i am showing you the graph of the confussion matrix. more the bright more erroneous it is. 
    so by getting the confussion matrix graph we canlearn from it and we can change the values of 
    classifier or by removing the noise from the images we can get the optimized result"""
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
########################################################################################################################

#Support vector classifier
clf_svm = LinearSVC()

#Training the model
clf_svm.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ("Linear SVM accuracy: ",acc_svm)
cross_val=cross_val_score(clf_svm, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_svm, X_train, y_train, cv=3)
print(y_train_pred)

#Confussion matrix
conf_mx=confusion_matrix(y_train, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print("Precision Score:",ps)
ps=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",ps)
"""Here i am showing you the graph of the confussion matrix. more the bright more erroneous it is. 
    so by getting the confussion matrix graph we canlearn from it and we can change the values of 
    classifier or by removing the noise from the images we can get the optimized result"""
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)

########################################################################################################################

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print ("nearest neighbors accuracy: ",acc_knn)
cross_val=cross_val_score(clf_knn, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_knn, X_train, y_train, cv=3)
print(y_train_pred)

#Confussion matrix
conf_mx=confusion_matrix(y_train, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print("Precision Score:",ps)
ps=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",ps)
"""Here i am showing you the graph of the confussion matrix. more the bright more erroneous it is. 
    so by getting the confussion matrix graph we canlearn from it and we can change the values of 
    classifier or by removing the noise from the images we can get the optimized result"""
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
