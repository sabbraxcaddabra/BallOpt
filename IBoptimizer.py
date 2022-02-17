from Ballistics.BallisticsClasses import *
from Optimization.Optimizers.RandomOptimizers import RandomSearchOptimizer
from Optimization.Constraints import *
import numpy as np

class BallOptimizer(BallisticsProblem):

    def __init__(self, barl, charge, shell, *args,
                 percentage=[10]*5, pmax_dop=np.inf, delta_max=np.inf, max_eta_k=np.inf, **kwargs):
        super().__init__(barl, charge, shell, *args, **kwargs)
        self.percentage = percentage
        self.pmax_dop = pmax_dop
        self.delta_max = delta_max
        self.max_eta_k = max_eta_k

    def set_bounds(self, percentage):
        shell_mass_bounds = Bounds((1 - percentage[0] / 100) * self.shell.q,
                            (1 + percentage[0] / 100) * self.shell.q)
        ld_bounds = (((1 - percentage[1] / 100) * self.barl.l_d,
                            (1 + percentage[1] / 100) * self.barl.l_d), )
        w0_bounds = (((1 - percentage[2] / 100) * self.barl.W0,
                            (1 + percentage[2] / 100) * self.barl.W0), )

        charge = self.charge
        om_bounds = [((1 - percentage[3] / 100) * powd.omega, (1 + percentage[3] / 100) * powd.omega) for powd in charge]
        jk_bounds = [((1 - percentage[4] / 100) * powd.Jk, (1 + percentage[4] / 100) * powd.Jk) for powd in charge]

        bounds_dict = {
            'shell_mass': shell_mass_bounds,
            'l_d': ld_bounds,
            'w0': w0_bounds,
            'om': om_bounds,
            'jk': jk_bounds
        }

        return bounds_dict

    def set_constrains(self):
        pmax_constraint = opt.NonlinearConstraint(
            lambda x: self.get_pmax(), lb=0, ub=self.pmax_dop
        )
        delta_constraint = opt.NonlinearConstraint(
            lambda x: self.count_delta(), lb=0, ub=self.delta_max
        )
        eta_k_constraint = opt.NonlinearConstraint(
            lambda x: self.get_eta_k(), lb=0, ub=self.max_eta_k
        )
        constraints = {
            'pmax':pmax_constraint,
            'delta':delta_constraint,
            'eta_k':eta_k_constraint
        }
        return constraints

    def count_delta(self):
        charge = self.ibproblem.charge
        return sum(powd.omega for powd in charge)

    def count_ib(self, x_vec, var_dict):

        for key, value in var_dict.items():
            self.adapters[key](x_vec, value)

        try:
            y, p_mean_max, p_sn_max, p_kn_max, psi_sum, eta_k = solve_ib(*self.ibproblem.create_params_tuple())
            self.pmax = p_mean_max
            self.eta_k = eta_k
            return -y[0]
        except:
            return 9999999999999

    def set_q(self, x, ind):
        self.ibproblem.syst.q = x[ind]

    def set_ld(self, x, ind):
        self.ibproblem.syst.l_d = x[ind]

    def set_w0(self, x, ind):
        self.ibproblem.syst.W0 = x[ind]

    def set_powd_mass(self, x, fst_ind):
        for i, powd in enumerate(self.ibproblem.charge, start=fst_ind):
            powd.omega = x[i]

    def set_Jk(self, x, fst_ind):
        for i, powd in enumerate(self.ibproblem.charge, start=fst_ind):
            powd.Jk = x[i]

    def get_powd_mass_vec(self):
        charge = self.ibproblem.charge
        powd_mass_vec = np.array([powd.omega for powd in charge])
        return powd_mass_vec

    def get_jk_vec(self):
        charge = self.ibproblem.charge
        jk_vec = np.array([powd.Jk for powd in charge])
        return jk_vec

    def get_ld(self):
        return self.ibproblem.syst.l_d

    def get_q(self):
        return self.ibproblem.syst.q

    def get_w0(self):
        return self.ibproblem.syst.w0

    def get_pmax(self):
        return self.pmax

    def get_eta_k(self):
        return self.eta_k

    def optimize(self, vars=['om', 'jk'], constrains=['pmax', 'delta']):
        x0_vec = np.array([])
        var_dict = dict()
        lubounds = []
        cons = []

        for var in vars:
            ind = len(x0_vec)
            var_dict[var] = ind
            lubounds.append(*self.x_bounds[var])

            x0_vec = np.append(x0_vec, self.variables[var]())

        for constrain in constrains:
            cons.append(self.constraints[constrain])

        opts = {
            'maxiter':len(x0_vec)*100
        }

        x_opt = opt.differential_evolution(self.count_ib, x0=x0_vec, args=(var_dict, ), bounds=lubounds,
                                           constraints=cons)
        print(x_opt)

    def out_func(self, x):
        pmax = self.get_pmax()*1e-6
        print(pmax, x, sep='\n')



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