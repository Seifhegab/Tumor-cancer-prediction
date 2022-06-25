from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# -----Logistic Regression-----
class LogisticReg:
    def __init__(self, X_train, X_test, Y_train, Y_test, Y_pred_log, log):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_pred_log = Y_pred_log
        self.log = log
    def logisticregression(self):
        # Define the Model
        # [Solver liblinear] we used it to can train in this small
        self.log = LogisticRegression(solver='liblinear')

        # Train the Model
        self.log.fit(self.X_train.values, self.Y_train.values)

        # Print the Training Accuracy score
        print('[1]Logistic Regression Training Accuracy score : ', self.log.score(self.X_train, self.Y_train))

        # Predict the response for test dataset
        self.Y_pred_log = self.log.predict(self.X_test.values)

        # confusion matrix
        print("confusion matrix of Logistic Regression:", '\n', confusion_matrix(self.Y_test, self.Y_pred_log))

        # Model Accuracy: how often is the classifier correct
        # Accuracy = (tp + tn) / (tp + tn + fn + fp)
        print("Accuracy of Logistic Regression:", metrics.accuracy_score(self.Y_test, self.Y_pred_log))

        # Model Precision: total number of all observations that have been predicted to belong
        # to the positive class and are actually positive
        # Precision = tp / (tp + fp)
        print("Precision of Logistic Regression:", metrics.precision_score(self.Y_test, self.Y_pred_log))

        # Model Recall: This is the proportion of observation predicted to belong to the positive
        # class, that truly belongs to the positive class.
        # Recall = tp / (tp + fn)
        print("Recall of Logistic Regression:", metrics.recall_score(self.Y_test, self.Y_pred_log))
        print('\n')

# -----------------------------------------------------------------------------------------------------
# -----Decision Tree-----
class DecisTree:
    def __init__(self, X_train, X_test, Y_train, Y_test, Y_predtree, tree):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_predtree = Y_predtree
        self.tree = tree

    def decisiontree(self):
        # Define the Model
        self.tree = DecisionTreeClassifier()

        # Train the Model
        self.tree.fit(self.X_train.values, self.Y_train.values)

        # Print the Training Accuracy score
        print('[2]Decision Tree Training Accuracy score : ', self.tree.score(self.X_train, self.Y_train))

        # Predict the response for test dataset
        self.Y_predtree = self.tree.predict(self.X_test.values)

        # confusion matrix
        print("confusion matrix of Decision Tree:", '\n', confusion_matrix(self.Y_test, self.Y_predtree))

        # Model Accuracy: how often is the classifier correct
        # Accuracy = (tp + tn) / (tp + tn + fn + fp)
        print("Accuracy of Decision Tree:", metrics.accuracy_score(self.Y_test, self.Y_predtree))

        # Model Precision: total number of all observations that have been predicted to belong
        # to the positive class and are actually positive
        # Precision = tp / (tp + fp)
        print("Precision of Decision Tree:", metrics.precision_score(self.Y_test, self.Y_predtree))

        # Model Recall: This is the proportion of observation predicted to belong to the positive
        # class, that truly belongs to the positive class.
        # Recall = tp / (tp + fn)
        print("Recall of Decision Tree:", metrics.recall_score(self.Y_test, self.Y_predtree))
        print('\n')

# -----------------------------------------------------------------------------------------------------
# -----Support Vector Machine (SVM)-----
class SVMclass:
    def __init__(self, X_train, X_test, Y_train, Y_test, Y_predsv, sv):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_predsv = Y_predsv
        self.sv = sv

    def svmfunc(self):
        # Define the Model
        self.sv = svm.SVC(kernel='linear')

        # Train the Model
        self.sv.fit(self.X_train.values, self.Y_train.values)

        # Print the Training Accuracy score
        print('[3]SVM Training Accuracy score by linear kernel : ', self.sv.score(self.X_train, self.Y_train))

        # self.sv = SVC(kernel='poly')
        # self.sv.fit(self.X_train, self.Y_train)
        # print('[2]SVM Training Accuracy by polynomial kernel: ', self.sv.score(self.X_train, self.Y_train))

        # self.sv = SVC(kernel='rbf')
        # self.sv.fit(self.X_train, self.Y_train)
        # print('[2]SVM Training Accuracy by Radial basis function kernel: ', self.sv.score(self.X_train, self.Y_train))

        # Predict the response for test dataset
        self.Y_predsv = self.sv.predict(self.X_test.values)

        # confusion matrix
        print("confusion matrix of SVM:", '\n', confusion_matrix(self.Y_test, self.Y_predsv))

        # Model Accuracy: how often is the classifier correct
        # Accuracy = (tp + tn) / (tp + tn + fn + fp)
        print("Accuracy of SVM:", metrics.accuracy_score(self.Y_test, self.Y_predsv))

        # Model Precision: total number of all observations that have been predicted to belong
        # to the positive class and are actually positive
        # Precision = tp / (tp + fp)
        print("Precision of SVM:", metrics.precision_score(self.Y_test, self.Y_predsv))

        # Model Recall: This is the proportion of observation predicted to belong to the positive
        # class, that truly belongs to the positive class.
        # Recall = tp / (tp + fn)
        print("Recall of SVM:", metrics.recall_score(self.Y_test, self.Y_predsv))
        print('\n')

# -----------------------------------------------------------------------------------------------------
# -----Random forest Classifier-----
class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test, Y_predforest, forest):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_predforest = Y_predforest
        self.forest = forest

    def randomforest(self):
        # Define the Model
        self.forest = RandomForestClassifier()

        # Train the Model
        self.forest.fit(self.X_train.values, self.Y_train.values)

        # Print the Training Accuracy score
        print('[4]Random forest Classifier Training Accuracy score : ', self.forest.score(self.X_train, self.Y_train))

        # Predict the response for test dataset
        self.Y_predforest = self.forest.predict(self.X_test.values)

        # confusion matrix
        print("confusion matrix of Random forest Classifier:", '\n', confusion_matrix(self.Y_test, self.Y_predforest))

        # Model Accuracy: how often is the classifier correct
        # Accuracy = (tp + tn) / (tp + tn + fn + fp)
        print("Accuracy of Random forest Classifier:", metrics.accuracy_score(self.Y_test, self.Y_predforest))

        # Model Precision: total number of all observations that have been predicted to belong
        # to the positive class and are actually positive
        # Precision = tp / (tp + fp)
        print("Precision of Random forest Classifier:", metrics.precision_score(self.Y_test, self.Y_predforest))

        # Model Recall: This is the proportion of observation predicted to belong to the positive
        # class, that truly belongs to the positive class.
        # Recall = tp / (tp + fn)
        print("Recall of Random forest Classifier:", metrics.recall_score(self.Y_test, self.Y_predforest))
        print('\n')
