from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from numpy import ndarray
import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.metrics import r2_score, mean_absolute_error
from regressors import stats
from scipy.stats import f


class Correlation:

    def __init__(self, data_frame: DataFrame, relevance_min_value: float):
        self.instance = data_frame.corr()
        self.relevance_min_value = relevance_min_value
        self.correlated_params = self.get_correlated_params()

    def get_correlated_params(self) -> list:
        fields = []
        params = []
        for i in self.instance:
            for j in self.instance.index[abs(self.instance[i]) > self.relevance_min_value]:
                if i != j and j not in fields and i not in fields:
                    fields.append(j)
                    params.append((i, j, self.instance[i][self.instance.index==j].values[0]))

        return params


class Regression:
    def __init__(self, data_frame: DataFrame, drop_params: list, correlation: Correlation):
        self.model = LinearRegression()
        self.correlation = correlation
        self.result_df = self._build_result_df(data_frame)
        self.result_nd = self.result_df.values.ravel()
        self.params_df = self._build_params_df(data_frame, drop_params)
        self.fit()
        self.predicted = self.predict()
        self.params_names = self.params_df.columns.values
        self.deviation = None
        self.std_errors = None
        self.t_values = None
        self.p_values = None
        self.f_stat = None
        self.dfn = self.params_df.shape[1]
        self.dfd = self.params_df.shape[0] - self.dfn - 1
        self.common_p_value = None
        self.f_crit = None
        self.r_squared = None
        self.adj_r_squared = None

    def calculate_metrics(self):
        self.deviation = self.calculate_deviation()
        self.std_errors = self.calculate_std_errors()
        self.t_values = self.calculate_t_values()
        self.p_values = self.calculate_p_values()
        self.f_stat = self.calculate_f_stat()
        self.common_p_value = self.calculate_common_p_value(self.f_stat)
        self.f_crit = self.calculate_f_crit()
        self.adj_r_squared = self.calculate_adj_r_squared()
        self.r_squared = self.calculate_r_squared()

    def calculate_r_squared(self):
        return r2_score(self.result_nd, self.predicted)

    def calculate_adj_r_squared(self):
        return stats.adj_r2_score(self.model, self.params_df, self.result_nd)

    def calculate_f_crit(self):
        return f.ppf(1-0.05, dfn=self.dfn, dfd=self.dfd)

    def calculate_common_p_value(self, f_stat):
        return f.sf(f_stat, self.dfn, self.dfd)

    def calculate_f_stat(self):
        return stats.f_stat(self.model, self.params_df, self.result_nd)

    def calculate_p_values(self):
        return stats.coef_pval(self.model, self.params_df, self.result_nd)

    def calculate_t_values(self):
        return stats.coef_tval(self.model, self.params_df, self.result_nd)

    def calculate_std_errors(self):
        return stats.coef_se(self.model, self.params_df, self.result_nd)

    def calculate_deviation(self):
        return mean_absolute_error(self.result_nd, self.predicted)

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
        print("Построение регрессионной модели")
        drop_params = self.read_drop_params()
        if len(drop_params) > 0:
            print("Параметры {} будут исключены из модели".format(','.join(drop_params)))

        return Regression(self.data_frame, drop_params, self.correlation)

    def read_drop_params(self) -> list:
        params = [x for x in self.data_frame.columns.values if x != 'y']
        print("Какие параметры исключить из модели? (перечислить через запятую, доступны {})".format(','.join(params)))
        drop = []
        for param in input().split(','):
            if param not in params:
                continue
            drop.append(param)

        return drop
