#************************************************ Bibliotheque*****************************************************************
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #for the model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.metrics import roc_curve #for model evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
#************************************************* Importing_Data ************************************************************
data = pd.read_csv('C:\\Users\\belar\\Desktop\\untitled\\heart.csv')
#print(data.corr())
dataX=data.drop('target',axis=1)
dataY=data['target']
X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.2,random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)
estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values
export_graphviz(estimator, out_file='tree.dot',
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True,
                label='root',
                precision = 2, filled = True)
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('somefile.png')
y_predict = model.predict(X_test)
print('Random_Forest',classification_report(y_test, y_predict)) # output accuracy
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred_bin)
confusion_matrix
total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# Logistic

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(X_train, y_train) # Train/Fit model
y_pred1 = model1.predict(X_test) # get y predictions
print('Logistic_Regression',classification_report(y_test, y_pred1)) # output accuracy

# KNN
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_i = knn.predict(X_test)
    print('Knn_iter_num',i,classification_report(y_test, y_pred_i))  # output accuracy