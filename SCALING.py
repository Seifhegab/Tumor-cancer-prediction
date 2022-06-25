from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns


class Scaling:
    def __init__(self, X):
        self.X = X

    def scale(self):
        # Data Scaling for data set
        # initialize normalizer
        data_norm = Normalizer()

        # Fit the data
        # Normalization formula(Z) = (X - Mean)/ Variance
        Normalize = data_norm.fit_transform(self.X)

        # Distribution plot
        # And we put with the normalization the standardization by put [kde = True]
        sns.displot(Normalize[:, 5], color='red', kde=True)

        # Add the axis labels
        plt.xlabel('patient data scaling')

        # Display the plot
        plt.show()