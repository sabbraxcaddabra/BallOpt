import matplotlib.pyplot as plt
from PyQt5.QtCore import QSysInfo

from InternalBallistics.ErrorClasses import *
from GUI.Analyze import analysisGUI                      #конвертированный в .py фал дизайна окна анализа
from PyQt5.QtGui import QFont, QCursor

from InternalBallistics.Analyze.SolveIntBal import solve_ib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtWidgets, Qt, QtCore, QtGui

from PyQt5.QtWidgets import QMessageBox, QHeaderView, QWidget


class IntBalAnalysis(QtCore.QObject):
    running = False
    finished = QtCore.pyqtSignal()
    result = QtCore.pyqtSignal(dict)

    too_much_powder_exception = QtCore.pyqtSignal()
    too_much_time_exception = QtCore.pyqtSignal()


    def __init__(self, parent, int_bal_cond):
        QtCore.QObject.__init__(self)
        self.parent = parent
        self.int_bal_cond = int_bal_cond

    def run(self):
        methods = {
            'Метод Рунге-Кутты 4-го порядка':'RK4',
            'Метод Адамса-Башфорда 5-го порядка':'AB5'
        }

        method_ = self.parent.method_comboBox.currentText()

        tau = float(self.parent.step_lineEdit.text())

        try:

            ts, ys, p_mean, p_sn, p_kn, lk_index = solve_ib(*self.int_bal_cond.create_params_tuple(), method=methods[method_], tstep=tau)

            current_result = {
                'ts':ts,
                'ys':ys,
                'p_mean':p_mean,
                'p_sn':p_sn,
                'p_kn':p_kn,
                'lk_index':lk_index
            }
            self.result.emit(current_result)

        except TooMuchPowderError:
            self.too_much_powder_exception.emit()

        except TooMuchTime:
            self.too_much_time_exception.emit()

        self.finished.emit()


# В этом классе прописываются все взаимодействия с окно АНАЛИЗА
class AnalysisApp(QtWidgets.QMainWindow, analysisGUI.Ui_AnalysWindow):   #Поменять название Ui_Dialog
    def __init__(self, parent=None, int_bal_cond=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        self.int_bal_cond = int_bal_cond
        self.first_row_text()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_Layout.addWidget(self.toolbar)
        self.plot_Layout.addWidget(self.canvas)
        self.current_result = None
        # Скрываем таблицу
        #self.result_table.hide()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Чиним таблицу с результатами
        self.FixTableWindows7_10(self.result_table)

        # Блокируем вкладку графика
        self.ResWindow.setTabEnabled(1, False)



        # События для кнопок
        self.butt_raschet.clicked.connect(self.do_raschet)
        self.plot_comboBox.view().pressed.connect(self.handleItemPressed)

        self.butt_close.clicked.connect(self.close)                         # Событие кнопки "Закрыть"

    # Метод чинит таблицу с результатам на Win10
    def FixTableWindows7_10(self, tableName):
        def SetStyle():
            tableName.setStyleSheet(
                "QHeaderView::section{"
                "border-top:0px solid #b9b9b9;"
                "border-left:0px solid #b9b9b9;"
                "border-right:1px solid #b9b9b9;"
                "border-bottom: 1px solid #b9b9b9;"
                "background-color:#fafafa;"
                "padding:4px;"
                "}"
                "QTableCornerButton::section{"
                "border-top:0px solid #b9b9b9;"
                "border-left:0px solid #b9b9b9;"
                "border-right:1px solid #b9b9b9;"
                "border-bottom: 1px solid #b9b9b9;"
                "background-color: #fafafa;"
                "}"
                "QTableWidget::item{"
                "color: black;"
                "selection-color: black;"
                #"selection-background-color: #fcfcfc;"
                "}"
            )

        if QSysInfo.windowsVersion() == QSysInfo.WV_WINDOWS10:
            SetStyle()

    # Метод проверяет введённые данные
    def CheckValue(self):

        # В случае нахождения будем вызывать диалоговое окно и передавать в него текст ошибки
        def ErrorDialog(textError):
            errorInit = QMessageBox()
            errorInit.setWindowTitle("Ошибка!")
            errorInit.setText(textError)
            errorInit.setIcon(QMessageBox.Critical)
            errorInit.setStandardButtons(QMessageBox.Cancel)
            buttCancel = errorInit.button(QMessageBox.Cancel)
            buttCancel.setText("Отмена")
            errorInit.exec()

        # Проверяем параметры заряжания
        Step = self.step_lineEdit.text()

        if not Step:
            errorText = "Укажите шаг расчёта!"
            ErrorDialog(errorText)
            return False

        # Меняем запятую на точку
        Step = Step.replace(",", ".")
        self.step_lineEdit.setText(Step)

        # Пытаемся значение ячейки преобразовать во float, если не получается - выводим диалоговое с ошибкой
        try:
            Step = float(Step)
            if Step <=0:
                errorText = "Шаг расчёта должен быть более нуля!"
                ErrorDialog(errorText)
                return False

        except ValueError:
            errorText = "Шаг расчёта задан некорректно!"
            ErrorDialog(errorText)
            return False

        # Если всё чики-пуки - возвращаем True
        return True

    def first_row_text(self):
        # Если порох один, то оставляем просто Пси, без индекса
        if len(self.int_bal_cond.charge) == 1:
            pass
        else:
            a = Qt.QTableWidgetItem(u"\u03C8"+"₁")

            # Ставим жирный шрифт у заголовка
            fontBold = QFont()
            fontBold.setBold(True)
            a.setFont(fontBold)
            self.result_table.setHorizontalHeaderItem(3, a)

            for i in range(1, len(self.int_bal_cond.charge)):

                #Делаем красивые индексы
                powder_num = str(i+1)
                SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                LowIndex = powder_num.translate(SUB)
                a = Qt.QTableWidgetItem(u"\u03C8"+LowIndex)

                # Ставим жирный шрифт у заголовков
                fontBold = QFont()
                fontBold.setBold(True)
                a.setFont(fontBold)

                self.result_table.insertColumn(3 + i)
                self.result_table.setHorizontalHeaderItem(3+i, a)


    def handleItemPressed(self, index):
        item = self.plot_comboBox.model().itemFromIndex(index)
        if item.text() == '<не указан>':
            pass
        else:
            self.plot_(item.text())

    def do_raschet(self):
        # Делаем проверку введённого значения шага расчёта
        if not self.CheckValue():
            return False
        # На время расчёт меняем курсор на задумчивый
        self.ResWindow.setCursor(QtCore.Qt.BusyCursor)




        # Перед новым расчётом скрываем старый листинг, блокируем вкладку графика
        ##self.result_table.hide()
        self.result_table.clearContents()
        self.plots.setDisabled(True)
        self.ResWindow.setTabEnabled(1, False)


        self.thread = QtCore.QThread()
        self.analysis = IntBalAnalysis(self, self.int_bal_cond)
        self.analysis.moveToThread(self.thread)

        self.analysis.result.connect(self.get_result)
        self.analysis.too_much_powder_exception.connect(self.raise_error_window_too_much_powder)
        self.analysis.too_much_time_exception.connect(self.raise_error_window_too_much_time)

        self.thread.started.connect(self.analysis.run)

        self.analysis.finished.connect(self.plot_)
        self.analysis.finished.connect(self.fill_average_pressure_and_gun_speed)
        self.analysis.finished.connect(self._fill_result_table)
        self.analysis.finished.connect(self.enable_plot)

        self.analysis.finished.connect(self.thread.quit)
        self.analysis.finished.connect(self.analysis.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # запустим поток
        self.thread.start()

    @QtCore.pyqtSlot()
    def enable_plot(self):

        # Делаем видимым вкладку график
        self.ResWindow.setTabEnabled(1, True)
        self.plots.setEnabled(True)
        self.plot_comboBox.setEnabled(True)
        # Удаляем пункт "Не указан"
        self.plot_comboBox.removeItem(self.plot_comboBox.findText('<не указан>'))

    @QtCore.pyqtSlot()
    def fill_average_pressure_and_gun_speed(self):
        if self.current_result:
            self.lineEdit_GunSpeed.setText(f'{self.current_result["ys"][0][-1]:.5g}')
            self.lineEdit_AverPress.setText(f'{max(self.current_result["p_mean"]) * 1e-6:.5g}')
        else:
            pass

    @QtCore.pyqtSlot(dict)
    def get_result(self, result_dict):
        self.current_result = result_dict

    @QtCore.pyqtSlot()
    def raise_error_window_too_much_powder(self):
        errorTooMuchPowd = QMessageBox()
        errorTooMuchPowd.setWindowTitle("Ошибка")
        errorTooMuchPowd.setText("Слишком много пороха!")
        errorTooMuchPowd.setIcon(QMessageBox.Critical)
        errorTooMuchPowd.setStandardButtons(QMessageBox.Cancel)

        buttCancel = errorTooMuchPowd.button(QMessageBox.Cancel)
        buttCancel.setText("Отмена")

        errorTooMuchPowd.exec()

    @QtCore.pyqtSlot()
    def raise_error_window_too_much_time(self):
        errorTooMuchTime = QMessageBox()
        errorTooMuchTime.setWindowTitle("Ошибка")
        errorTooMuchTime.setText("Превышено время выстрела (1с)!")
        errorTooMuchTime.setIcon(QMessageBox.Critical)
        errorTooMuchTime.setStandardButtons(QMessageBox.Cancel)

        buttCancel = errorTooMuchTime.button(QMessageBox.Cancel)
        buttCancel.setText("Отмена")

        errorTooMuchTime.exec()

    @QtCore.pyqtSlot()
    def plot_(self, grafics_dict_key='Среднебаллистическое давление'):
        plot_dict = {
            'Среднебаллистическое давление': (self._pressure_graphics, ('p_mean', 'Среднебаллистическое давление')),
            'Давление на дно снаряда': (self._pressure_graphics, ('p_sn', 'Давление на дно снаряда')),
            'Давление на дно канала ствола': (self._pressure_graphics, ('p_kn', 'Давление на дно канала ствола')),
            'Скорость снаряда': (self._velocity_graphic, ('Скорость снаряда',))
        }
        if self.current_result:
            plot_dict[grafics_dict_key][0](*plot_dict[grafics_dict_key][1])

    def _pressure_graphics(self, vals='p_mean', title='Среднебаллистическое давление'):
        pressure = self.current_result[vals] * 1e-6
        l_d = self.current_result['ys'][1]
        # instead of ax.hold(False)
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        # ax.hold(False) # deprecated, see above
        # plot data
        ax.plot(l_d, pressure)
        # max_index = np.argmax(pressure)
        # ax.scatter(l_d[max_index], pressure[max_index])
        lk_index = self.current_result['lk_index']
        if lk_index:
            ax.scatter(l_d[lk_index], pressure[lk_index], linewidths=3, label=f"Окончание горения заряда(lk = {round(l_d[lk_index], 3)} м)")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Координата по стволу Ld, м')
        ax.set_ylabel('Давление P, МПа')
        ax.set_xlim(right=1.1 * l_d[-1])
        ax.set_ylim(top=1.1 * max(pressure))
        ax.grid()

        # refresh canvas
        self.canvas.draw()
    def _velocity_graphic(self, title='Скорость снаряда'):
        velocity = self.current_result['ys'][0]
        l_d = self.current_result['ys'][1]

        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        # ax.hold(False) # deprecated, see above
        # plot data
        ax.plot(l_d, velocity)

        lk_index = self.current_result['lk_index']
        if lk_index:
            ax.scatter(l_d[lk_index], velocity[lk_index], label=f"Окончание горения заряда(lk = {round(l_d[lk_index], 3)} м)")
            ax.legend()

        ax.set_title(title)
        ax.set_xlim(right=1.1 * l_d[-1])
        ax.set_ylim(top=1.1 * max(velocity))
        ax.set_xlabel('Координата по стволу Ld, м')
        ax.set_ylabel('Скорость V, м/с')
        ax.grid()

        self.canvas.draw()

    @QtCore.pyqtSlot()
    def _fill_result_table(self):
        if self.current_result:


            #Вставляем строку если расчёт первый
            if self.result_table.rowCount() == 0:
                self.result_table.setRowCount(1)
            # Очищаем старые данные
            valRow = self.result_table.rowCount()


            if valRow > 1:
                for i in range(valRow, 0 , -1):
                    self.result_table.removeRow(i)
            """
            Заполнение таблицы результатов
            :return:
            """


            for timestep in range(len(self.current_result['ts'])):
                self.result_table.insertRow(timestep+1)
                a = Qt.QTableWidgetItem(str(round(self.current_result['ts'][timestep]*1e3, 2)))
                a.setTextAlignment(QtCore.Qt.AlignCenter)
                a.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.result_table.setItem(timestep, 0, a)
                for index, y in enumerate(self.current_result['ys'], start=1):
                    a = Qt.QTableWidgetItem(str(round(y[timestep], 2)))
                    a.setTextAlignment(QtCore.Qt.AlignCenter)
                    a.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # Отключаем возможность редакт. ячейку
                    self.result_table.setItem(timestep, index, a)
                p_map = map(self.current_result.get, ('p_mean', 'p_sn', 'p_kn'))

                for i, p in enumerate(p_map, start=index+1):
                    a = Qt.QTableWidgetItem(str(round(p[timestep]*1e-6, 2)))
                    a.setTextAlignment(QtCore.Qt.AlignCenter)
                    a.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    self.result_table.setItem(timestep, i, a)
            # Удаляем лишнюю строку
            valRow = self.result_table.rowCount()
            self.result_table.removeRow(valRow-1)

            # Настраиваем ширину заголовков
            # Узнаем текущее количество колонок
            ColVal = self.result_table.columnCount()
            header = self.result_table.horizontalHeader()

            for col in range(ColVal):
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            # Меняем курсор обратно
            self.ResWindow.setCursor(QtCore.Qt.ArrowCursor)
        else:
            pass