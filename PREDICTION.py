class Prediction:
    def __init__(self, Y_pred_log, Y_predtree, Y_predsv, Y_predforest):
        self.Y_pred_log = Y_pred_log
        self.Y_predtree = Y_predtree
        self.Y_predsv = Y_predsv
        self.Y_predforest = Y_predforest

    def predict(self):
        # Print the Prediction (Logistic Regression)
        print("prediction of Logistic Regression", "\n", self.Y_pred_log)
        print('\n')

        # Print the Prediction (Decision Tree)
        print("prediction of Decision Tree", "\n", self.Y_predtree)
        print('\n')

        # Print the Prediction (SVM)
        print("prediction of SVM", "\n", self.Y_predsv)
        print('\n')

        # Print the Prediction (Random forest Classifier)
        print("prediction of Random forest Classifier", "\n", self.Y_predforest)
        print('\n')

