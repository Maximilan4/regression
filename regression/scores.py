import numpy as np
from regression.analyze import Regression
from numpy import ndarray
from pandas import DataFrame, Series, concat
from sklearn.metrics import r2_score, max_error
from sklearn.feature_selection import f_regression
from regressors import stats


class RegressionScore:

    def __init__(self, regression: Regression):
        self.regression = regression

    def print(self):
        print('Отчет:')
        print('-----------------')

        xlabels = self.regression.params_df.columns.values
        stats.summary(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel(), xlabels)

    def get_main_data_frame(self) -> DataFrame:
        return DataFrame([[
            stats.coef_se(self.regression.model, self.regression.params_df, self.regression.result_df.values.ravel())
        ]], columns=['standart error', 'max_errors'])

    def r2(self, predicted_values: ndarray):
        return r2_score(self.regression.result_df, predicted_values)

    def max_errors(self, predicted_values: ndarray):
        return max_error(self.regression.result_df, predicted_values)

    def f_test(self):
        f_test, p_values = f_regression(self.regression.params_df, self.regression.result_df.values.ravel())
        return f_test / np.max(f_test), p_values

    def print_prediction_with_info(self, values: list):
        x_values = np.array([values])
        x_series = Series(values, index=self.regression.params_df.columns.values)
        prediction = self.regression.predict(x_values)
        print('Значения параметров:')
        print(x_series)
        print('Значение y: {}'.format(prediction))

    def get_params_data_frame(self) -> DataFrame:
        f_test, p_values = self.f_test()
        param_ser = Series(self.regression.params_df.columns.values, name='param')
        f_test_ser = Series(f_test, name='f-test')
        p_values_ser = Series(p_values, name='p-value')
        coef_ser = Series(self.regression.model.coef_, name='coef')

        df = concat([param_ser, coef_ser, f_test_ser, p_values_ser], axis=1)

        return df.set_index('param')
