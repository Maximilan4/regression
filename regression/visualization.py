import matplotlib.pyplot as plt
from regression.analyze import Regression


class RegressionPlot:

    def __init__(self, regression: Regression):
        self.regression = regression

    def create(self):
        plt.scatter(self.regression.params_df, self.regression.result_df)
        plt.plot(self.regression.params_df, self.regression.predicted, color='red')
        plt.show()
