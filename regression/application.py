import os

from .analyze import RegressionAnalyzer, Regression
from .serializers import RegressionStdOutSerializer


class Application:

    def __init__(self, sources_path):
        self.sources_path = sources_path
        self.file_path = None

    def run(self):
        self.setup_file()
        self.run_analyse()

    def run_analyse(self):
        analyzer = RegressionAnalyzer(self.file_path)
        regression = analyzer.run()
        regression.calculate_metrics()
        serializer = RegressionStdOutSerializer(regression)
        serializer.serialize()
        values = self.get_to_predict_values(regression)
        serializer.print_prediction_with_info(values)

    def get_to_predict_values(self, regression: Regression) -> list:
        print('Создание прогноза')
        print('Вбейте значения x через запятую для получения прогнозного значения y')
        x_columns = regression.params_names
        print(','.join(x_columns))
        values = [float(v) for v in input().split(',')]
        if len(values) == 0 or len(values) < len(x_columns):
            print('Не вбито ни одного значения, или вбито меньшее количество параметров')

        return values

    def setup_file(self):
        files = [file for file in os.listdir(self.sources_path) if file.endswith('.csv')]
        for index, file in enumerate(files, 0):
            print('[{}] - {}'.format(index, file))

        inserted_index = int(input('Введите индекс файла: '))
        try:
            self.file_path = os.path.join(self.sources_path, files[inserted_index])
        except IndexError:
            print('Не могу найти файл. Попробуйте еще')
            self.setup_file()
