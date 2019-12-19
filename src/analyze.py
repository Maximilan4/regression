from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from numpy import ndarray
from sklearn.metrics import mean_absolute_error


class Regression:
    def __init__(self, data_frame: DataFrame, drop_params: list):
        self.model = LinearRegression()
        self.result_df = self._build_result_df(data_frame)
        self.params_df = self._build_params_df(data_frame, drop_params)
        self.fit()
        self.predicted = self.predict()
        self.deviation = self.calculate_deviation(self.result_df.values.ravel(), self.predicted)


    def calculate_deviation(self, orig_y, predicted_y):
        return mean_absolute_error(orig_y, predicted_y)

    @staticmethod
    def _build_params_df(data_frame: DataFrame, drop_params: list = None):
        if drop_params is None:
            drop_params = ['y']
        else:
            drop_params.append('y')

        return data_frame.drop(drop_params, axis=1)

    @staticmethod
    def _build_result_df(data_frame: DataFrame) -> DataFrame:
        return data_frame[['y']]

    def fit(self, x=None, y=None, weight=None):
        if x is not None and y is not None:
            self.model.fit(x, y, weight)

        self.model.fit(self.params_df, self.result_df.values.ravel())

    def predict(self, x=None) -> ndarray:
        if x is not None:
            values = self.model.predict(x)
        else:
            values = self.model.predict(self.params_df)

        return values


class RegressionAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_frame = self.read_file()
        self.correlation = None

    def read_file(self) -> DataFrame:
        return read_csv(self.file_path, ';')

    def print_data_frame(self):
        print(self.data_frame)

    def run(self) -> Regression:
        print("Загруженные данные:")
        self.print_data_frame()
        print("Построение кореляции:")
        relevance_value = float(input("Считать коллинеарными параметры больше:"))
        self.correlation = Correlation(self.data_frame, relevance_value)
        self.correlation.print()
        self.correlation.print_correlated_params()
        print("Построение регрессионной модели")
        drop_params = self.read_drop_params()
        if len(drop_params) > 0:
            print("Параметры {} будут исключены из модели".format(','.join(drop_params)))

        return Regression(self.data_frame, drop_params)

    def read_drop_params(self) -> list:
        params = [x for x in self.data_frame.columns.values if x != 'y']
        print("Какие параметры исключить из модели? (перечислить через запятую, доступны {})".format(','.join(params)))
        drop = []
        for param in input().split(','):
            if param not in params:
                continue
            drop.append(param)

        return drop


class Correlation:

    def __init__(self, data_frame: DataFrame, relevance_min_value: float):
        self.instance = data_frame.corr()
        self.relevance_min_value = relevance_min_value

    def print(self):
        print(self.instance)

    def print_correlated_params(self):
        fields = []
        for i in self.instance:
            for j in self.instance.index[self.instance[i] > self.relevance_min_value]:
                if i != j and j not in fields and i not in fields:
                    fields.append(j)
                    print("%s-->%s: r^2=%f" % (i, j, self.instance[i][self.instance.index==j].values[0]))
