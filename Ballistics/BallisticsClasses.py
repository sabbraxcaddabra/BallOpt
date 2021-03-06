from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from .calculations import internal_ballistics as ib
from .calculations import external_ballistics as eb

__all__ = ["ArtSystem", "Powder", "LoadParams",
           "ShootingParameters", "Shell", "BallisticsProblem"]


@dataclass
class ArtSystem:
    # Датакласс для данных об артиллерийской системе
    name: str  # Наименование артиллерийской системы
    d: float  # Приведенная площадь канала ствола
    S: float  # Калибр орудия
    W0: float  # Объем зарядной каморы
    l_d: float  # Полный путь снаряда
    l_k: float  # Длина зарядной каморы
    l0: float  # Приведенная длина зарядной каморы
    Kf: float  # Коэффициент слухоцкого

    def __str__(self):
        return f"Арт.система {self.name}, калибр: {self.d*1e3} мм"

@dataclass
class Shell:
    name:str # Индекс снаряда
    d:float # Калибр
    q:float # Масса снаряда
    i43:float # Коэф формы по закону 43 года
    alpha:float = 0. # Коэффициент наполнения

@dataclass
class Powder:
    # Датакласс для данных о порохе
    name: str  # Марка пороха
    omega: float  # Масса метательного заряда
    rho: float  # Плотность пороха
    f_powd: float  # Сила пороха
    Ti: float  # Температура горения пороха
    Jk: float  # Конечный импульс пороховых газов
    alpha: float  # Коволюм
    teta: float  # Параметр расширения
    Zk: float  # Относительная толщина горящего свода, соответствующая концу горения
    #PsiS: float # Относительная масса сгоревшего пороха к моменту распада
    kappa1: float  # 1-я, 2-я и 3-я хар-ки формы пороховых элементов до распада
    lambd1: float
    mu1: float
    kappa2: float  # 1-я, 2-я и 3-я характеристики формы пороховых элементов после распада
    lambd2: float
    mu2: float
    gamma_f: float # Температурная поправка на силу пороха
    gamma_Jk: float # Температурная поправка на конечный импульс

    def __str__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk*1e-3} кПа*с"

    def __repr__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk*1e-3} кПа*с"

    @classmethod
    def from_data_string(cls, string: str):
        string_list = string.strip().split(' ')
        data_list = list(map(float, string_list[1:]))
        return cls(string_list[0], 0.0, *data_list)

class LoadParams:
    # Класс хранящий информацию о параметрах заряжания
    def __init__(self, P0, T0=15., PV=None, ig_mass=None):
        self.P0 = P0  # Давление форсирования
        self.T0 = T0 # Температура метательного заряда
        if ig_mass:
            self.ig_mass = ig_mass # Масса воспламенителя
        else:
            self.PV = PV-1e5 # Давление воспламенителя

@dataclass
class ShootingParameters:
    theta_angle:float = 45. # Угол стрельбы
    distance:float = 150e3 # Макс. дистанция стрельбы


class BallisticsProblem:
    v0 = 0.
    pmax = 1e5
    psi_sum = 0.
    eta_k = 0.

    igniter_f = 240e3
    igniter_teta = 0.22
    igniter_Ti = 2427.

    Igniter = namedtuple('Igniter', [
        'fs',
        'num',
        'denum'
    ])
    Powder_ = namedtuple('Powd', [
        'omega',
        'rho',
        'f_powd',
        'Ti',
        'Jk',
        'alpha',
        'teta',
        'Zk',
        'kappa1',
        'lambd1',
        'mu1',
        'kappa2',
        'lambd2',
        'mu2'
    ])

    def __init__(self, barl, charge, shell, load_params=LoadParams(30e6, PV=4e5), shot_params=ShootingParameters()):
        self.barl = barl # Орудие
        self.charge = charge # Массив порохов(метательный заряд)
        self.shell = shell # Снаряд
        self.load_params = load_params # Параметры заряжания(внутреннаяя баллистика)
        self.shot_params = shot_params # Параметры стральбы(внешняя баллистика)

    def _ib_preprocessor(self):
        if not hasattr(self, 'igniter_mass'):
            self.igniter_mass = self.load_params.PV * (self.barl.W0 - sum(powd.omega / powd.rho for powd in self.charge)) / self.igniter_f

        fs = self.igniter_mass*self.igniter_f
        num = fs/self.igniter_Ti
        denum = num/self.igniter_teta

        igniter = self.Igniter(fs, num, denum)

        # Метод для создания исходных данных
        params = [
            self.load_params.P0,
            igniter,
            50e6 ** 0.25,
            self.barl.S,
            self.barl.W0,
            self.barl.l_k,
            self.barl.l0,
            sum(powd.omega for powd in self.charge),
            self.barl.Kf * self.shell.q,
            self.barl.l_d
        ]
        powders = []
        for powder in self.charge:
            tmp = self.Powder_(
                powder.omega,
                powder.rho,
                powder.f_powd * (1. + powder.gamma_f * (self.load_params.T0 - 15.)),
                powder.Ti,
                powder.Jk * (1. - powder.gamma_Jk * (self.load_params.T0 - 15.)),
                powder.alpha,
                powder.teta,
                powder.Zk,
                powder.kappa1,
                powder.lambd1,
                powder.mu1,
                powder.kappa2,
                powder.lambd2,
                powder.mu2
            )
            powders.append(tmp)
        params.append(tuple(powders))
        return tuple(params)

    def solve_ib(self, tstep=1e-5, tmax=1.):

        v0, p_mean_max, _, _, psi_sum, eta_k = ib.count_ib(*self._ib_preprocessor(), tstep=tstep, tmax=tmax)

        self.v0 = v0
        self.pmax = p_mean_max
        self.psi_sum = psi_sum
        self.eta_k = eta_k

        return v0, p_mean_max, psi_sum, eta_k

    def _eb_preprocessor(self):
        return (self.v0, self.shell.q, self.shell.d, self.shell.i43,
               np.deg2rad(self.shot_params.theta_angle), self.shot_params.distance)

    def solve_eb(self, tstep=1., tmax=1000.):
        x_max, y_end, v_end, theta_end = eb.count_eb(*self._eb_preprocessor(), tstep, tmax)
        self.Lmax = x_max
        self.y_end = y_end
        self.v_end = v_end
        self.theta_end = theta_end
        return x_max, y_end, v_end, theta_end

if __name__ == '__main__':
    artsys = ArtSystem(name='2А42', d=.03, S=0.000735299, W0=0.125E-3, l_d=2.263, l_k=0.12,
                       l0=0.125E-3 / 0.000735299, Kf=1.136)
    shell = Shell('30ка', 0.03, 0.389, 1.)

    powders = [Powder(name='6/7', omega=0.12, rho=1.6e3, f_powd=988e3, Ti=2800., Jk=343.8e3, alpha=1.038e-3, teta=0.236,
               Zk=1.53, kappa1=0.239, lambd1=2.26, mu1=0., kappa2=0.835, lambd2=-0.943, mu2=0., gamma_f=3e-4,
               gamma_Jk=0.0016)]

    bal_prob = BallisticsProblem(
        artsys, powders, shell,
        shot_params=ShootingParameters(5., 1000.)
    )
    print(bal_prob.solve_ib())
    print(bal_prob.solve_eb())