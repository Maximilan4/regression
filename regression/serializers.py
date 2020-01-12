import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from .analyze import Regression, Correlation
from pandas import Series


class Serializer(metaclass=ABCMeta):

    @abstractmethod
    def serialize(self, *args, **kwargs):
        pass


class RegressionSerializer(Serializer):

    def __init__(self, regression: Regression):
        self.regression = regression


class RegressionDictSerializer(RegressionSerializer):

    def serialize(self, *args, **kwargs):
        return {
            'correlation': {
                'table': self.regression.correlation.instance.to_dict(),
                'relevance_min_value': self.regression.correlation.relevance_min_value,
                'correlated_params': self.regression.correlation.correlated_params
            },
            'predicted_y': self.regression.predicted.tolist(),
            'deviation': self.regression.deviation,
            'r_squared': self.regression.r_squared,
            'adj_r_squared': self.regression.adj_r_squared,
            'result-table': self.get_result_dict(),
            't-significance': self.get_significant_params_by_t(),
            'p-significance': self.get_significant_params_by_p(),
            'f-stat-values': self.get_f_stat_value()
        }

    def get_f_stat_value(self):
        return {
            'f-stat': self.regression.f_stat,
            'f-crit': self.regression.f_crit,
            'f-significance': self.regression.common_p_value,
            'is_zero_hypothesis': bool(self.regression.f_crit > self.regression.f_stat)
        }

    def get_significant_params_by_p(self):
        params = []
        p_values = self.regression.p_values
        p_values = np.delete(p_values, 0)

        for index, p in enumerate(p_values.tolist()):
            if p < 0.05:
                params.append(self.regression.params_names[index])

        return params

    def get_significant_params_by_t(self):
        params = []
        t_values = self.regression.t_values
        t_values = np.delete(t_values, 0)

        for index, t in enumerate(t_values.tolist()):
            if t > 2:
                params.append(self.regression.params_names[index])

        return params

    def get_result_dict(self):
        coef_df = pd.DataFrame(
            index=['y-zero'] + list(self.regression.params_names),
            columns=['estimates', 'std-error', 't-values', 'p-values']
        )

        coef_df['estimates'] = np.concatenate(
            (np.round(np.array([self.regression.model.intercept_]), 6),
             np.round(self.regression.model.coef_, 6))
        )
        coef_df['std-error'] = self.regression.std_errors
        coef_df['t-values'] = self.regression.t_values
        coef_df['p-values'] = self.regression.p_values

        return coef_df.to_dict()


class RegressionStdOutSerializer(RegressionSerializer):

    def serialize(self, *args, **kwargs):
        print('Отчет:')
        print('Таблица корреляции:')
        self.print_correlation_data(self.regression.correlation)
        print('\n')
        self.print_predicted_y()
        self.print_deviation()
        self.print_r_squared()
        print('\n')
        self.print_result_table()
        print('\n')
        self.print_t_value_result()
        print('\n')
        self.print_p_value_result()
        print('\n')
        self.print_f_value_results()

    def print_prediction_with_info(self, values: list):
        x_values = np.array([values])
        x_series = Series(values, index=self.regression.params_names)
        prediction = self.regression.predict(x_values)
        print('Значения параметров:')
        print(x_series)
        print('Значение y: {}'.format(prediction))

    def print_f_value_results(self):
        print("F статистика: {}".format(self.regression.f_stat))
        print("Значимость F: {}".format(self.regression.common_p_value))
        print("F кр: {}".format(self.regression.f_crit))

        if self.regression.f_crit > self.regression.f_stat:
            print("{} > {}, коэфициент детерминации не значим, нельзя отбросить нулевую гипотезу".format(
                self.regression.f_crit, self.regression.f_stat
            ))
        else:
            print("{} < {}, коэфициент детерминации значим, можно отбросить нулевую гипотезу".format(
                self.regression.f_crit, self.regression.f_stat
            ))

    def print_result_table(self):
        print('Результаты анализа:')
        coef_df = pd.DataFrame(
            index=['y-пересечение'] + list(self.regression.params_names),
            columns=['коэфициенты', 'стд. ошибка', 't знач', 'p знач']
        )

        coef_df['коэфициенты'] = np.concatenate(
            (np.round(np.array([self.regression.model.intercept_]), 6),
             np.round(self.regression.model.coef_, 6))
        )
        coef_df['стд. ошибка'] = self.regression.std_errors
        coef_df['t знач'] = self.regression.t_values
        coef_df['p знач'] = self.regression.p_values
        print(coef_df)

    def print_r_squared(self):
        print("Значение R^2: {}".format(self.regression.r_squared))
        print("Значение сглаженного R^2: {}".format(self.regression.adj_r_squared))

    def print_p_value_result(self):
        print("Определение значимости факторов исходя из значений p:")
        params = []
        p_values = self.regression.p_values
        p_values = np.delete(p_values, 0)

        for index, p in enumerate(p_values.tolist()):
            if p < 0.05:
                params.append(self.regression.params_names[index])

        if len(params) != 0:
            print("Статистически значимыми для модели являются параметры {}, т.к их p < 0.05".format(",".join(params)))
        else:
            print("Нет значимых для модели параметров")

    def print_t_value_result(self):
        print("Определение значимости факторов исходя из значений t:")
        params = []
        t_values = self.regression.t_values
        t_values = np.delete(t_values, 0)

        for index, t in enumerate(t_values.tolist()):
            if t > 2:
                params.append(self.regression.params_names[index])

        if len(params) != 0:
            print("Значимыми для модели являются параметры {}, т.к их t > 2".format(",".join(params)))
        else:
            print("Нет значимых для модели параметров")

    @staticmethod
    def print_correlation_data(correlation: Correlation):
        print(correlation.instance)
        print("Уровень значимости: {}".format(correlation.relevance_min_value))

        if len(correlation.correlated_params) != 0:
            print('Между собой коррелируют параметры:')
            for params in correlation.correlated_params:
                print("{}-->{}: r^2={}".format(*params))

    def print_predicted_y(self):
        print("Прогнозируемые значения y от текущей выборки параметров x: {}".format(self.regression.predicted))

    def print_deviation(self):
        print("Среднее абсолютное отклонение: {}".format(self.regression.deviation))
