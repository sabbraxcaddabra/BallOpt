import numpy as np
from math import floor
from Optimization.OptimizationErrors import *
from PyQt5.QtWidgets import QMessageBox

from GUI.Optimize import optimizGUI                      #конвертированный в .py фал дизайна окна анализа
from PyQt5 import QtWidgets, Qt, QtCore, QtGui
from PyQt5.QtCore import QThread
import time

from InternalBallistics.Optimize.SolveIntBal import solve_ib
from InternalBallistics.Optimize.IntBalOptimizer import IntBalOptimizer
from InternalBallistics.ErrorClasses import *
from InternalBallistics.ErrorClasses import *


class Optimization(QtCore.QObject):
    running = False
    finished = QtCore.pyqtSignal()
    new_info = QtCore.pyqtSignal(str)
    progress_bar_updater = QtCore.pyqtSignal(int)
    progress_bar_updater_sel_comp = QtCore.pyqtSignal(int)
    error_signal = QtCore.pyqtSignal(object)

    counter = 0

    def __init__(self, parent):
        QtCore.QObject.__init__(self)
        self.parent = parent



    def run(self):
        methods = {
            'Случайный поиск': 'random_search',
            'Случайное сканирование': 'random_scan'
        }

        #method = methods[self.parent.comboBox_MethOptimize.currentText()]

        self.int_bal_cond = self.parent.parent.set_int_bal_cond()

        x_vec = []

        for powd in self.int_bal_cond.charge:
            x_vec.extend((powd.omega, powd.Jk))

        x_vec = np.array(x_vec)

        #combo_index = self.parent.comboBox_MethOptimize.currentIndex()

        #if combo_index == 0:
        x_lims = [[0, np.inf] for _ in range(len(x_vec))]
        # else:
        #     powd_mass_lim = float(self.parent.val_massPowd.text())/100.
        #
        #     finit_imp_lim = float(self.parent.val_FinitImpuls.text())/100
        #
        #     powd_mass_lims = [[0., powd.omega + powd_mass_lim*powd.omega] for powd in self.int_bal_cond.charge]
        #     Jk_lims = [[powd.Jk - finit_imp_lim*powd.Jk, powd.Jk + finit_imp_lim*powd.Jk] for powd in self.int_bal_cond.charge]
        #     x_lims = []
        #     for lim in zip(powd_mass_lims, Jk_lims):
        #         x_lims.extend(lim)



        max_delta = float(self.parent.val_maxDensity.text())

        p_max = float(self.parent.val_maxPress.text()) * 1e6

        if self.parent.checkBox_regGor.isChecked():
            max_eta_k = float(self.parent.val__coordGor.text())
            optimizer = IntBalOptimizer(x_vec, params=self.int_bal_cond, Pmax=p_max,
                                        max_eta_k=max_eta_k, delta_max=max_delta, x_lims=x_lims)
        else:
            optimizer = IntBalOptimizer(x_vec, params=self.int_bal_cond, Pmax=p_max,
                                        delta_max=max_delta, x_lims=x_lims)


        if self.parent.checkBox_SelComp.isChecked():

            residuals = float(self.parent.val__resid.text())
            optimizer.out_func = None
            try:
                optimized_xvec, optimized_f, optimized_sol, optimized_summary = optimizer.optimize_with_Jk()
                optimizer._adapt(optimized_xvec)
                text = 'Лучший вариант\n'
                for powd in optimizer.params.charge:
                    text += f"Масса пороха {powd.name}: {round(powd.omega, 4)} кг\n"
                    text += f"Конечный импульс пороха {powd.name}: {round(powd.Jk*1e-3, 2)} кПа*с\n"
                text += f"Целевая функция {round(optimized_f, 1)}\n"
                text += f"Дульная скорость: {round(optimized_sol[0], 1)} м/с\n"
                text += f"Максимальное среднебаллистическое давление: {round(optimized_sol[1] * 1e-6, 2)} МПа\n"
                text += f"Максимальное давление на дно снаряда: {round(optimized_sol[2] * 1e-6, 2)} МПа\n"
                text += f"Максимальное давление на дно канала ствола: {round(optimized_sol[3] * 1e-6, 2)} МПа\n"
                text += f"Относительная масса сгоревшего пороха {round(optimized_sol[4], 2)} \n"
                text += f"Относительная координата полного сгорания порохового заряда {round(optimized_sol[5], 3)}\n"
                self.new_info.emit(text)
                self.pick_up_optimum_charge(optimizer, optimized_xvec, residuals)
            except Exception as E:
                self.error_signal.emit(E)
        else:
            try:
                optimizer.out_func = self.out_func
                optimized_xvec = optimizer.optimize_with_Jk()[0]

            except FirstStepOptimizationFail as E:
                self.error_signal.emit(E)

            except MinStepOptimizerError as E:
                self.error_signal.emit(E)
            except:
                print('Всё пошло не так как хотелось бы. Однако нужно ли это и то ли это что должно было ли быть?! Вопрос с философсикм подходм личного моего мнения у меня нет')




        self.finished.emit()

    def combo_info(self, num, info_dict):
        text = f'Комбинация № {num}\n'
        for powd in info_dict['combo']:
            text += f"Масса пороха {powd.name}: {round(powd.omega, 4)} кг\n"
            text += f"Конечный импульс пороха {powd.name}: {round(powd.Jk*1e-3, 2)} кПа*с\n"
        text += f"Целевая функция {round(info_dict['target_func'], 1)}\n"
        text += f"Дульная скорость: {round(info_dict['sol'][0], 1)} м/с\n"
        text += f"Максимальное среднебаллистическое давление: {round(info_dict['sol'][1] * 1e-6, 2)} МПа\n"
        text += f"Относительная масса сгоревшего пороха {round(info_dict['sol'][4], 2)} \n"
        if info_dict['sol'][-1] != 0:
            text += f"Относительная координата полного сгорания порохового заряда {round(info_dict['sol'][5], 3)}\n"
        else:
            text += f"Координата полного сгорания порохового заряда: заряд не догорел\n"
        self.new_info.emit(text)


    def pick_up_optimum_charge(self, optimizer, optimized_xvec, max_tol):

        # self.parent.textBrowser_optimize.clear()

        optimizer._adapt(optimized_xvec)
        omega_sum = sum(powder.omega for powder in optimizer.params.charge)

        if "Jk" in optimizer.adapters.keys():
            optimizer.remove_adapter('Jk')

        optimizer.x_lims = optimizer.x_lims[::2]
        optimizer.out_func = None

        Jk_dop_list = [powd.Jk for powd in optimizer.params.charge]

        combos = optimizer.get_powder_combination(Jk_dop_list, max_tol)

        self.new_info.emit(f'\nНайдено {len(combos)} комбинаций порохов\n')

        optimized_combos = []
        n_combos = len(combos)

        for num, combo in enumerate(combos, start=1):
            try:
                info_dict = optimizer.optimize_one_charge(omega_sum, combo, 'random_search')
                optimized_combos.append(info_dict)
                self.combo_info(num, info_dict)
            except FirstStepOptimizationFail or MinStepOptimizerError as E:
                self.new_info.emit(f'Не найдено ни одного оптимума {str(combo)}')
                self.new_info.emit(str(E))

            self.progress_bar_updater_sel_comp.emit(floor((num/n_combos) * 100))

        best = min(optimized_combos, key=lambda info_dict: info_dict['target_func'])

        text = f'\nЛучший вариант\n'
        for powd in best['combo']:
            text += f"Масса пороха {powd.name}: {round(powd.omega, 4)} кг\n"
            text += f"Конечный импульс пороха {powd.name}: {round(powd.Jk*1e-3, 2)} кПа*с\n"
        text += f"Целевая функция {round(best['target_func'], 1)}\n"
        text += f"Дульная скорость: {round(best['sol'][0], 1)} м/с\n"
        text += f"Максимальное среднебаллистическое давление: {round(best['sol'][1] * 1e-6, 2)} МПа\n"
        text += f"Относительная масса сгоревшего пороха {round(best['sol'][4], 2)} \n"
        if best['sol'][-1] != 0:
            text += f"Относительная координата полного сгорания порохового заряда {round(best['sol'][-1], 3)}\n"
        else:
            text += f"Координата полного сгорания порохового заряда: заряд не догорел\n"
        self.new_info.emit(text)
        #self.new_info.emit('Расчет окончен')

    def out_func(self, x_vec, f, sol, params):
        text = ''
        for powd in params.charge:
            text += f"Масса пороха {powd.name}: {round(powd.omega, 4)} кг\n"
            text += f"Конечный импульс пороха {powd.name}: {round(powd.Jk*1e-3, 2)} кПа*с\n"
        text += f"Целевая функция {round(f, 1)}\n"
        text += f"Дульная скорость: {round(sol[0], 1)} м/с\n"
        text += f"Максимальное среднебаллистическое давление: {round(sol[1] * 1e-6, 2)} МПа\n"
        text += f"Максимальное давление на дно снаряда: {round(sol[2] * 1e-6, 2)} МПа\n"
        text += f"Максимальное давление на дно канала ствола: {round(sol[3] * 1e-6, 2)} МПа\n"
        text += f"Относительная масса сгоревшего пороха {round(sol[4], 2)}\n"
        text += f"Относительная координата полного сгорания заряда {round(sol[5], 3)}\n"
        text += "_" * 53 + '\n'
        self.counter += 1
        self.new_info.emit(text)
        self.progress_bar_updater.emit(self.counter)
        QtCore.QThread.msleep(100)
        return text


# В этом классе прописываются все взаимодействия с окном ОПТИМИЗАЦИИ
class OptimizeApp(QtWidgets.QMainWindow, optimizGUI.Ui_OptimizeWindow):   #Поменять название Ui_Dialog
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        # Скрываем прогресс бар и лейбл
        self.progressBar_procOptimize.hide()
        self.label_procOptimize.hide()

        self.butt_Start.clicked.connect(self.do_optimize)

        #self.comboBox_MethOptimize.view().pressed.connect(self.handleItemPressed)
        self.checkBox_regGor.stateChanged.connect(self.checkRegGor)
        self.checkBox_SelComp.stateChanged.connect(self.checkRegResid)
        self.butt_close.clicked.connect(self.close)

    #Метод учёта догорания заряда
    def checkRegGor(self):
        if self.checkBox_regGor.isChecked():
            self.val__coordGor.clear()
            self.label_coordGor.setEnabled(True)
            self.val__coordGor.setEnabled(True)
        else:
            self.val__coordGor.clear()
            self.label_coordGor.setDisabled(True)
            self.val__coordGor.setDisabled(True)



    #Метод учета невязки по конечному импульсу
    def checkRegResid(self):
        if self.checkBox_SelComp.isChecked():
            self.val__resid.clear()
            self.label_resid.setEnabled(True)
            self.val__resid.setEnabled(True)
            self.val__resid.setText("15")
        else:
            self.val__resid.clear()
            self.label_resid.setDisabled(True)
            self.val__resid.setDisabled(True)

    # Вызов окна ошибки
    @QtCore.pyqtSlot(object)
    def ErrorWindow(self, E):
        self.textBrowser_optimize.clear()
        E = str(E)
        errorOptimize = QMessageBox()
        errorOptimize.setWindowTitle("Ошибка в процессе оптимизации")
        errorOptimize.setText(E)
        errorOptimize.setIcon(QMessageBox.Critical)
        errorOptimize.setStandardButtons(QMessageBox.Cancel)
        buttCancel = errorOptimize.button(QMessageBox.Cancel)
        buttCancel.setText("Отмена")
        errorOptimize.exec()

    # def handleItemPressed(self, index):
    #     item = self.comboBox_MethOptimize.model().itemFromIndex(index)
    #     if item.row() == 1:
    #         self.label_massPowd.setEnabled(True)
    #         self.label_FinitImpuls.setEnabled(True)
    #         self.val_massPowd.setEnabled(True)
    #         self.val_FinitImpuls.setEnabled(True)
    #
    #     if item.row() == 0:
    #         self.label_massPowd.setDisabled(True)
    #         self.label_FinitImpuls.setDisabled(True)
    #         self.val_massPowd.setDisabled(True)
    #         self.val_FinitImpuls.setDisabled(True)
    #         self.val_massPowd.clear()
    #         self.val_FinitImpuls.clear()

    # Метод выполняет проверку всех введённых данных
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


        # Проверяем данные ограничений 1-го рода:

        maxDensity = self.val_maxDensity.text()
        if not maxDensity:
            errorText = "Укажите все ограничения 1-го рода!"
            ErrorDialog(errorText)
            return False
        # Меняем запятую на точку
        maxDensity = maxDensity.replace(",", ".")
        self.val_maxDensity.setText(maxDensity)

        # Пытаемся значение ячейки преобразовать во float, если не получается - выводим диалоговое с ошибкой
        try:
            maxDensity = float(maxDensity)
            if maxDensity <= 0:
                errorText = "Максимальная плотность заряжания должна быть больше нуля!"
                ErrorDialog(errorText)
                return False

        except ValueError:
            errorText = "Ограничения 1-го рода заданны некорректно!"
            ErrorDialog(errorText)
            return False

        if self.checkBox_SelComp.isChecked():


            resid = self.val__resid.text()
            if not resid:
                errorText = "Укажите значение невязки!"
                ErrorDialog(errorText)
                return False

            # Меняем запятую на точку
            resid = resid.replace(",", ".")
            self.val__resid.setText(resid)


            # Пытаемся значение ячейки преобразовать во float, если не получается - выводим диалоговое с ошибкой
            try:
                resid = float(resid)
                if (resid > 100) or (resid <= 0):
                    errorText = "Значение невязки должно быть более 0% и менее 100%!"
                    ErrorDialog(errorText)
                    return False


            except ValueError:
                errorText = "Значение невязки заданно некорректно!"
                ErrorDialog(errorText)
                return False



        # Проверяем данные ограничений 2-го рода:

        maxPress = self.val_maxPress.text()
        if not maxPress:
            errorText = "Укажите все ограничения 2-го рода!"
            ErrorDialog(errorText)
            return False
        # Меняем запятую на точку
        maxPress = maxPress.replace(",", ".")
        self.val_maxPress.setText(maxPress)
        # Пытаемся значение ячейки преобразовать во float, если не получается - выводим диалоговое с ошибкой
        try:
            maxPress = float(maxPress)
            if maxPress <= 0:
                errorText = "Максимальное давление должно быть более нуля!"
                ErrorDialog(errorText)
                return False

        except ValueError:
            errorText = "Ограничения 2-го рода заданны некорректно!"
            ErrorDialog(errorText)
            return False

        if self.checkBox_regGor.isChecked():
            coordGor = self.val__coordGor.text()
            if not coordGor:
                errorText = "Укажите все ограничения 2-го рода!"
                ErrorDialog(errorText)
                return False
            # Меняем запятую на точку
            coordGor = coordGor.replace(",", ".")
            self.val__coordGor.setText(coordGor)
            # Пытаемся значение ячейки преобразовать во float, если не получается - выводим диалоговое с ошибкой
            try:
                coordGor = float(coordGor)
                if (coordGor > 1) or (coordGor <=0 ):
                    errorText = "Значение относ. координаты должно быть более 0 и менее 1!"
                    ErrorDialog(errorText)
                    return False

            except ValueError:
                errorText = "Ограничения 2-го рода заданны некорректно!"
                ErrorDialog(errorText)
                return False

        return True



    def do_optimize(self):
        # Проверка данных
        if not self.CheckValue():
            return False
        # Чистим листинг
        self.textBrowser_optimize.clear()

        # Показываем прогресс бар и лейбл на время расчёта
        self.label_procOptimize.show()
        self.progressBar_procOptimize.show()


        # создадим поток
        self.thread = QtCore.QThread()
        # создадим объект для выполнения кода в другом потоке
        self.browserHandler = Optimization(self)
        # перенесём объект в другой поток
        self.browserHandler.moveToThread(self.thread)
        # после чего подключим все сигналы и слоты
        self.browserHandler.new_info.connect(self.add_new_text)
        self.browserHandler.progress_bar_updater.connect(self.update_progress_bar)
        self.browserHandler.progress_bar_updater_sel_comp.connect(self.update_progress_bar_select_components)
        self.browserHandler.error_signal.connect(self.ErrorWindow)

        # подключим сигнал старта потока к методу run у объекта, который должен выполнять код в другом потоке
        self.thread.started.connect(self.browserHandler.run)
        # self.browserHandler.finished.connect(lambda: self.progressBar_procOptimize.setValue(100))
        # self.browserHandler.finished.connect(lambda: time.sleep(0.5))
        # self.browserHandler.finished.connect(lambda: self.progressBar_procOptimize.hide())
        # self.browserHandler.finished.connect(lambda: self.label_procOptimize.hide())
        self.browserHandler.finished.connect(self.hide_progress_bar)
        self.browserHandler.finished.connect(self.thread.quit)
        self.browserHandler.finished.connect(self.browserHandler.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)
        # запустим поток
        self.thread.start()


    @QtCore.pyqtSlot()
    def hide_progress_bar(self):
        self.progressBar_procOptimize.setValue(100)
        time.sleep(0.5)
        self.progressBar_procOptimize.setValue(0)
        self.progressBar_procOptimize.hide()
        self.label_procOptimize.hide()


    @QtCore.pyqtSlot(str)
    def add_new_text(self, string):
        self.textBrowser_optimize.append(string)

    @QtCore.pyqtSlot(int)
    def update_progress_bar(self, counter):
        tmp = self.progressBar_procOptimize.value()
        if tmp <= 90:
            ProgrVal = tmp + np.random.randint(3, 6)*counter
            self.progressBar_procOptimize.setValue(ProgrVal)

    @QtCore.pyqtSlot(int)
    def update_progress_bar_select_components(self, counter):
        self.progressBar_procOptimize.setValue(counter)
