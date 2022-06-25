from PRE_PROCESS import Process
from sklearn.preprocessing import LabelEncoder
from ALGORITHMS import LogisticReg
from ALGORITHMS import DecisTree
from ALGORITHMS import SVMclass
from ALGORITHMS import RandomForest
from PREDICTION import Prediction
from VOTING import Voting
from SCALING import Scaling
import pandas as pd
import joblib

Tumor = pd.read_csv("Tumor Cancer Prediction_Data.csv")

X = Y = X_train = X_test = Y_train = Y_test = Y_pred_log = Y_predtree = Y_predsv = Y_predforest = log = tree = sv = \
    forest = 0

# calling preprocessing function
p1 = Process(Tumor, X, Y, X_train, X_test, Y_train, Y_test)
p1.pre_process()
print(p1.Tumor)

# calling Logistic Regression algorithm to train the module
l1 = LogisticReg(p1.X_train, p1.X_test, p1.Y_train, p1.Y_test, Y_pred_log, log)
l1.logisticregression()

# calling Decision Tree algorithm to train the module
d1 = DecisTree(p1.X_train, p1.X_test, p1.Y_train, p1.Y_test, Y_predtree, tree)
d1.decisiontree()

# calling Support Vector Machine (SVM) algorithm to train the module
s1 = SVMclass(p1.X_train, p1.X_test, p1.Y_train, p1.Y_test, Y_predsv, sv)
s1.svmfunc()

# calling Random forest Classifier algorithm to train the module
r1 = RandomForest(p1.X_train, p1.X_test, p1.Y_train, p1.Y_test, Y_predforest, forest)
r1.randomforest()

# calling Prediction of four algorithm
pr1 = Prediction(l1.Y_pred_log, d1.Y_predtree, s1.Y_predsv, r1.Y_predforest)
pr1.predict()

# calling Voting function
v1 = Voting(l1.Y_pred_log, d1.Y_predtree, s1.Y_predsv)
v1.vote()

# calling the scaling function
sc1 = Scaling(p1.X)
sc1.scale()

# -----------------------------------------------------------------------------------------------------
# Read players data
Tumor_test = pd.read_csv("Test Data 1.csv")

# filling all null values by 0
Tumor_test.fillna(0, inplace=True)

# change (diagnosis) column to (0 and 1) instead of (B and M) to be able to normalize the data
lb = LabelEncoder()
Tumor_test.iloc[:, 31] = lb.fit_transform(Tumor_test.iloc[:, 31].values)

# Split the dataset into independent(X) and dependent(Y) datasets
X = Tumor_test.iloc[:, 1:31]
Y = Tumor_test.iloc[:, 31]

# drop all duplicate rows
X.drop_duplicates(inplace=True)

p1.X_train = X.values
Y = Y.values

print(Tumor_test)
# -----------------------------------------------------------------------------------------------------

# Saving and Loading Models

# Save Model to file in the current working directory (Logistic Regression)
Save_file1 = 'Tumor_Cancer_Prediction by logistic regression.sav'
joblib.dump(l1.log, Save_file1)

# Save Model to file in the current working directory (Decision Tree)
Save_file2 = 'Tumor_Cancer_Prediction by Decision Tree.sav'
joblib.dump(d1.tree, Save_file2)

# Save Model to file in the current working directory (SVM)
Save_file3 = 'Tumor_Cancer_Prediction by svm.sav'
joblib.dump(s1.sv, Save_file3)

# Save Model to file in the current working directory (Random forest Classifier)
Save_file4 = 'Tumor_Cancer_Prediction by Random forest Classifier.sav'
joblib.dump(r1.forest, Save_file4)

# Load from file (Logistic Regression)
Load_file1 = joblib.load(Save_file1)

# Load from file (Decision Tree)
Load_file2 = joblib.load(Save_file2)

# Load from file (SVM)
Load_file3 = joblib.load(Save_file3)

# Load from file (Random forest Classifier)
Load_file4 = joblib.load(Save_file4)

# -----------------------------------------------------------------------------------------------------
# calling Prediction of four algorithm
pr2 = Prediction(Load_file1.predict(p1.X_train), Load_file2.predict(p1.X_train), Load_file3.predict(p1.X_train),
                 Load_file4.predict(p1.X_train))
pr2.predict()

# calling Voting function
v2 = Voting(Load_file1.predict(p1.X_train), Load_file2.predict(p1.X_train), Load_file3.predict(p1.X_train))
v2.vote()

# calling the scaling function
sc2 = Scaling(p1.X_train)
sc2.scale()
