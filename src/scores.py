import numpy as np
from src.analyze import Regression
from numpy import ndarray
from pandas import Series
from sklearn.metrics import r2_score
from regressors import stats
from scipy.stats import f


class RegressionScore:

    def __init__(self, regression: Regression):
        self.regression = regression
        self.params_names = self.regression.params_df.columns.values

    def print(self):
        print('Отчет:')
        print('-----------------')

        stats.summary(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel(),
                      self.params_names)
        self.print_predicted_y()
        self.print_deviation()
        print('-----------------')
        self.print_t_value_result()
        print('-----------------')
        self.print_p_value_result()
        print('-----------------')
        self.print_summary_p_value_result()

    def print_summary_p_value_result(self):
        f_value = stats.f_stat(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel())
        dfn = len(self.params_names)
        dfd = self.regression.params_df.shape[0] - len(self.params_names) - 1
        p_value = f.sf(f_value, dfn, dfd)
        print("Значимость F: {}".format(p_value))
        f_crit = f.ppf(1-0.05, dfn=dfn, dfd=dfd)
        print("F кр: {}".format(f_crit))

        if f_crit > f_value:
            print("{} > {}, коэфициент детерминации не значим, нельзя отбросить нулевую гипотезу".format(
                f_crit, f_value
            ))
        else:
            print("{} < {}, коэфициент детерминации значим, можно отбросить нулевую гипотезу".format(
                f_crit, f_value
            ))

    def print_p_value_result(self):
        p_values = stats.coef_pval(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel())
        print("Определение значимости факторов исходя из значений p:")
        params = []
        p_values = np.delete(p_values, 0)

        for index, p in enumerate(p_values.tolist()):
            if p < 0.05:
                params.append(self.params_names[index])

        if len(params) != 0:
            print("Статистически значимыми для модели являются параметры {}, т.к их p < 0.05".format(",".join(params)))
        else:
            print("Нет значимых для модели параметров")

    def print_t_value_result(self):
        t_values = stats.coef_tval(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel())
        print("Определение значимости факторов исходя из значений t:")
        params = []
        t_values = np.delete(t_values, 0)

        for index, t in enumerate(t_values.tolist()):
            if t > 2:
                params.append(self.params_names[index])

        if len(params) != 0:
            print("Значимыми для модели являются параметры {}, т.к их t > 2".format(",".join(params)))
        else:
            print("Нет значимых для модели параметров")

    def print_predicted_y(self):
        print("Прогнозируемые значения y от текущей выборки параметров x: {}".format(self.regression.predicted))

    def print_deviation(self):
        print("Среднее абсолютное отклонение: {}".format(self.regression.deviation))

    def r2(self, predicted_values: ndarray):
        return r2_score(self.regression.result_df, predicted_values)

    def print_prediction_with_info(self, values: list):
        x_values = np.array([values])
        x_series = Series(values, index=self.regression.params_df.columns.values)
        prediction = self.regression.predict(x_values)
        print('Значения параметров:')
        print(x_series)
        print('Значение y: {}'.format(prediction))
