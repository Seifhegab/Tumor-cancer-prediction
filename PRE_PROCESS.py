import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Process:
    def __init__(self, Tumor, X, Y, X_train, X_test, Y_train, Y_test):
        self.Tumor = Tumor
        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def pre_process(self):
        # filling all null values by 0
        self.Tumor.fillna(0, inplace=True)

        # change (diagnosis) column to (0 and 1) instead of (B and M) to be able to normalize the data
        lb = LabelEncoder()
        self.Tumor.iloc[:, 31] = lb.fit_transform(self.Tumor.iloc[:, 31].values)

        # Split the dataset into independent(X) and dependent(Y) datasets
        self.X = self.Tumor.iloc[:, 1:31]
        self.Y = self.Tumor.iloc[:, 31]

        # drop all duplicate rows
        self.X.drop_duplicates(inplace=True)

        # Split the dataset into 75% training and 25% testing
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=0)

        # standardize the data values
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train))
        self.X_test = pd.DataFrame(scaler.transform(self.X_test))
        print('\n\n')

