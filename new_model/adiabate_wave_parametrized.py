from mesh_t_2 import Mesh
import numpy as np
import matplotlib.pyplot as plt
from progonka import Progonka
from datetime import datetime
import os


# x,v,p,ρ,t,e,W,g,K,ω
class AdiabatWave:
    def __init__(self, N, T, tau, H, a_s, a_t, sigma=0.5, max_k_iters=10, nu=0.05, experiment_prefix='',
                 save_plots=False, save_params=False, show_plots=False,
                 ρ_L=1, ρ_R=1, V_L=-2, V_R=2, P_L=0.4, P_R=0.4, R=1, gamma=5. / 3, x_left=None, x_right=None
                 ):
        """

        :param N:
        :param T:
        :param tau:
        :param H:
        :param a_s:
        :param a_t:
        :param sigma:
        :param max_k_iters:
        :param nu:
        :param experiment_prefix:
        :param save_plots:
        :param save_params:
        :param show_plots:
        :param ρ_L:
        :param ρ_R:
        :param V_L:
        :param V_R:
        :param P_L:
        :param P_R:
        :param R:
        :param gamma:
        :param x_left:  граница визуализации по x
        :param x_right: граница визуализации по x
        """
        self.experiment_prefix = experiment_prefix
        self.experiment_name = f"{experiment_prefix} {datetime.now().strftime('%m-%d-%Y %H-%M-%S')}"
        self.save_plots = save_plots
        self.save_params = save_params
        self.show_plots = show_plots
        self.x_left = x_left
        self.x_right = x_right

        self.results_dir = os.path.join(os.path.dirname(__file__), self.experiment_name)
        if os.path.exists(self.results_dir):
            pass
        else:
            os.mkdir(self.results_dir)

        self.N = N
        self.T = T
        self.a_s = a_s
        self.a_t = a_t
        self.tau = tau
        self.H = H

        # ρ = 1 плотность
        self.ρ_L = ρ_L
        self.ρ_R = ρ_R
        # (self.gamma + 1)/(self.gamma - 1)

        self.V_L = V_L  # 2./(self.gamma+1)
        self.V_R = V_R

        self.P_L = P_L
        self.P_R = P_R

        self.R = R

        self.t_R = P_R / (ρ_R * self.R)
        self.t_L = P_L / (ρ_L * self.R)  # 2 * (self.gamma - 1) / ((self.gamma + 1) ** 2)

        self.v = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='v')
        self.x = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='x')
        self.p = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='p')
        self.ρ = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='ρ')
        self.e = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='e')
        self.t = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='t')
        self.g = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='g')
        self.ω = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='ω')
        self.K = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='K')
        self.W = Mesh(N=N, T=T, tau=tau, h=self.H, a_s=a_s, a_t=a_t, name='W')

        # self.R = 8.31
        # self.gamma = 4. / 3

        self.sigma = sigma
        self.sigma_1 = 0.5
        self.sigma_2 = 0.5

        self.max_k_iters = max_k_iters  # максимальное число итераций

        self.c = 0.5  # страница 216

        self.epsilon1 = 0.01
        self.epsilon2 = 0.01

        # переметр искуственной вязкости (вместо ν)
        self.nu = nu

        # # параметры ударной волны
        self.gamma = gamma

        self.ρ_k = None
        self.v_k = None
        self.x_k = None
        self.p_l = None
        self.e_l = None
        self.t_l = None
        self.ω_l = None
        self.W_l = None
        self.K_l = None

    def h(self, i):
        if (i == -1) or (i == self.N):
            return self.H  # 0
        else:
            return self.H

    def h_dash(self, i):
        """h со штрихом для полуцелых интервалов"""
        return 0.5 * (self.h(i) + self.h(i - 1))

    def plot_all(self):
        # итоговые графики
        self.v.plot_mesh(save_path=self.results_dir + '/v_mesh.png', show_plot=self.show_plots)
        self.p.plot_mesh(save_path=self.results_dir + '/p_mesh.png', show_plot=self.show_plots)
        self.ρ.plot_mesh(save_path=self.results_dir + '/ρ_mesh.png', show_plot=self.show_plots)
        self.x.plot_mesh(save_path=self.results_dir + '/x_mesh.png', show_plot=self.show_plots)
        self.e.plot_mesh(save_path=self.results_dir + '/e_mesh.png', show_plot=self.show_plots)
        self.t.plot_mesh(save_path=self.results_dir + '/t_mesh.png', show_plot=self.show_plots)

        if self.T >= 20:
            t_list = range(0, self.T, int(self.T / 5))
        else:
            t_list = [0, 1, 2, 3, 4, 5]

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.v._mesh[t_moment, self.v.good_points()], label=str(t_moment))
        plt.title('V')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/v.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.p._mesh[t_moment, self.p.good_points()], label=str(t_moment))
        plt.title('P')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/P.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.ρ._mesh[t_moment, self.ρ.good_points()], label=str(t_moment))
        plt.title('ρ')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/ρ.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.x._mesh[t_moment, self.x.good_points()], label=str(t_moment))
        plt.title('x')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/x.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.t._mesh[t_moment, self.t.good_points()], label=str(t_moment))
        plt.title('t')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/t.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            plt.plot(self.t._mesh[t_moment, self.e.good_points()], label=str(t_moment))
        plt.title('E')
        plt.xlabel('i')
        plt.ylabel('j')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/E.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(self.x._mesh[:, int(1)], label='левая граница ударника')
        plt.plot(self.x._mesh[:, int(int(self.N / 2) + 1)], label='правая граница ударника')

        plt.plot(self.x._mesh[:, int((int(self.N / 2) + 1) + 1)], label='левая граница мишени')
        plt.plot(self.x._mesh[:, int(self.N + 1)], label='правая граница мишени')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/Ударник и мишень.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(self.x._mesh[:, int(int(self.N / 2) + 1)] - self.x._mesh[:, int(1)],
                 label='изменение размеров ударника')
        plt.plot(self.x._mesh[:, int(self.N + 1)] - self.x._mesh[:, int((int(self.N / 2) + 1) + 1)],
                 label='изменение размеров мишени')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/изменение размеров ударника и мишени.png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            if (not self.x_left is None) and (not self.x_right is None):
                x_list = []
                for _x in self.x.good_points():
                    if self.x_left <= self.x._mesh[t_moment, _x] <= self.x_right:
                        x_list.append(_x)
            else:
                x_list = self.x.good_points()
            plt.plot(self.x._mesh[t_moment, x_list], self.ρ._mesh[t_moment, x_list],
                     label=f"t = {t_moment * self.tau}")
        plt.title('ρ(x)')
        plt.xlabel('x')
        plt.ylabel('ρ')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/ρ(x).png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            if (not self.x_left is None) and (not self.x_right is None):
                x_list = []
                for _x in self.x.good_points():
                    if self.x_left <= self.x._mesh[t_moment, _x] <= self.x_right:
                        x_list.append(_x)
            else:
                x_list = self.x.good_points()
            plt.plot(self.x._mesh[t_moment, x_list], self.v._mesh[t_moment, x_list],
                     label=f"t = {t_moment * self.tau}")

        plt.title('V(x)')
        plt.xlabel('x')
        plt.ylabel('V')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/V(x).png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            if (not self.x_left is None) and (not self.x_right is None):
                x_list = []
                for _x in self.x.good_points():
                    if self.x_left <= self.x._mesh[t_moment, _x] <= self.x_right:
                        x_list.append(_x)
            else:
                x_list = self.x.good_points()
            plt.plot(self.x._mesh[t_moment, x_list], self.p._mesh[t_moment, x_list],
                     label=f"t = {t_moment * self.tau}")

        plt.xlabel('x')
        plt.ylabel('P')
        plt.title('P(x)')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/P(x).png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            if (not self.x_left is None) and (not self.x_right is None):
                x_list = []
                for _x in self.x.good_points():
                    if self.x_left <= self.x._mesh[t_moment, _x] <= self.x_right:
                        x_list.append(_x)
            else:
                x_list = self.x.good_points()
            plt.plot(self.x._mesh[t_moment, x_list], self.e._mesh[t_moment, x_list],
                     label=f"t = {t_moment * self.tau}")
        plt.title('E(x)')
        plt.xlabel('x')
        plt.ylabel('E')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/E(x).png')
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(10, 10))
        for t_moment in t_list:
            if (not self.x_left is None) and (not self.x_right is None):
                x_list = []
                for _x in self.x.good_points():
                    if self.x_left <= self.x._mesh[t_moment, _x] <= self.x_right:
                        x_list.append(_x)
            else:
                x_list = self.x.good_points()
            plt.plot(self.x._mesh[t_moment, x_list], self.t._mesh[t_moment, x_list],
                     label=f"t = {t_moment * self.tau}")
        plt.title('t(x)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        if self.save_plots:
            plt.savefig(self.results_dir + '/t(x).png')
        if self.show_plots:
            plt.show()

    def run(self):
        self.set_initial_conditions()
        for _j in range(self.T):
            self.calculate_layer_j(_j)
        self.plot_all()

    def func_P(self, ρ, T):
        return self.R * T * ρ

    def func_მP_by_მρ(self, T):
        return self.R * T

    def func_მP_by_მT(self, ρ):
        return self.R * ρ

    def func_E(self, T, ρ=None):
        return self.R * T / (self.gamma - 1)

    def func_მE_by_მρ(self):
        return 0.

    def func_მE_by_მT(self):
        return self.R / (self.gamma - 1)

    def func_K(self, T):
        # if T==0:
        #     return 0
        # return 0.01*(T**0.5  )
        return 0.

    def func_მK_by_მρ(self):
        return 0.

    def func_მK_by_მρ_prev(self):
        return 0.

    def func_მK_by_მT(self, T):

        # if T==0:
        #     return 0
        # return  0.01* 0.5*(T**(-0.5))
        return 0.

    def func_მK_by_მT_prev(self):
        return 0.

    def func_Ω(self, v, v_next, ρ, h_dash):
        """
        функция искусственной вязкости
        :return:
        """
        return -0.5 * self.nu * ρ * ((v_next - v) / h_dash - np.abs((v_next - v) / h_dash))

    def func_მΩ_by_მρ(self, v, v_next, ρ, h_dash):
        return self.func_Ω(v, v_next, ρ, h_dash) / ρ

    def func_მΩ_by_მv(self, v, v_next, ρ, h_dash):
        v_s = (v_next - v) / h_dash
        if v_s < 0:
            return self.nu * ρ / h_dash
        else:
            return 0.

    def func_g(self, v, v_next, ρ, T, h_dash):
        return self.func_P(ρ, T) + self.func_Ω(v, v_next, ρ, h_dash)

    def func_მg_by_მρ(self, v, v_next, ρ, T, h_dash):
        return self.func_მP_by_მρ(T) + self.func_მΩ_by_მρ(v, v_next, ρ, h_dash)

    def func_W(self, t, t_prev, h_dash):
        return 0  # -self.func_K(t) * (t - t_prev) / h_dash

    def set_initial_conditions(self):
        """
        скорость страница 229
        v(0,t) = V1
        v(M,t) = V2
        :return:
        """

        if self.save_params:
            with open(self.results_dir + '/params.txt', 'w') as f:
                f.write(f"T= {self.T}\n")
                f.write(f"N= {self.N}\n")
                f.write(f"tau= {self.tau}\n")
                f.write(f"h= {self.H}\n")
                f.write(f"sigma= {self.sigma}\n")
                f.write(f"max_k_iters= {self.max_k_iters}\n")
                f.write(f"gamma= {self.gamma}\n")
                f.write(f"density_R= {self.ρ_R}\n")
                f.write(f"density_L= {self.ρ_L}\n")
                f.write(f"t_R= {self.t_R}\n")
                f.write(f"t_L= {self.t_L}\n")
                f.write(f"V_R= {self.V_R}\n")
                f.write(f"V_L= {self.V_L}\n")

        bound = int(self.N / 2)
        for __x in range(-1, bound):
            self.v.set_i_j(__x, 0, self.V_L)
            self.ρ.set_i_j(__x, 0, self.ρ_L)
            self.t.set_i_j(__x, 0, self.t_L)

        for __x in range(bound, self.N + 2):
            self.v.set_i_j(__x, 0, self.V_R)
            self.ρ.set_i_j(__x, 0, self.ρ_R)
            self.t.set_i_j(__x, 0, self.t_R)

        for i in range(0, self.N + 1):
            _x = self.h(i) / (self.ρ.get_i_j(i, 0)) + self.x.get_i_j(i, 0)
            self.x.set_i_j(i + 1, 0, _x)

        # x,v,p,ρ,t,e,W,g,K,ω
        for i in range(-1, self.N + 2):
            self.p.set_i_j(i, 0, self.func_P(self.ρ.get_i_j(i, 0), self.t.get_i_j(i, 0)))
            self.e.set_i_j(i, 0, self.func_E(self.t.get_i_j(i, 0), self.ρ.get_i_j(i, 0)))
            self.K.set_i_j(i, 0, self.func_K(self.t.get_i_j(i, 0)))

        for i in range(0, self.N + 2):
            self.W.set_i_j(i, 0, self.func_W(self.t.get_i_j(i, 0), self.t.get_i_j(i - 1, 0), self.h_dash(i)))

        for i in range(0, self.N + 1):
            self.ω.set_i_j(i, 0, self.func_Ω(self.v.get_i_j(i, 0),
                                             self.v.get_i_j(i + 1, 0),
                                             self.ρ.get_i_j(i, 0),
                                             self.h_dash(i)))

            self.g.set_i_j(i, 0, self.func_g(self.v.get_i_j(i, 0),
                                             self.v.get_i_j(i + 1, 0),
                                             self.ρ.get_i_j(i, 0),
                                             self.t.get_i_j(i, 0),
                                             self.h_dash(i)
                                             ))

    def get_f1(self, i, k, j):
        gk = (self.p_k.get_i_j(i, k) + self.func_Ω(self.v_k.get_i_j(i, k),
                                                   self.v_k.get_i_j(i + 1, k),
                                                   self.ρ_k.get_i_j(i, k),
                                                   self.h_dash(i)
                                                   )) - \
             (self.p_k.get_i_j(i - 1, k) + self.func_Ω(self.v_k.get_i_j(i - 1, k),
                                                       self.v_k.get_i_j(i, k),
                                                       self.ρ_k.get_i_j(i - 1, k),
                                                       self.h_dash(i)))

        gj = (self.p.get_i_j(i, j) + self.func_Ω(self.v.get_i_j(i, j),
                                                 self.v.get_i_j(i + 1, j),
                                                 self.ρ.get_i_j(i, j),
                                                 self.h_dash(i))) - \
             (self.p.get_i_j(i - 1, j) + self.func_Ω(self.v.get_i_j(i - 1, j),
                                                     self.v.get_i_j(i, j),
                                                     self.ρ.get_i_j(i - 1, j),
                                                     self.h_dash(i)
                                                     ))

        f1_k_i = self.v_k.get_i_j(i, k) - self.v.get_i_j(i, j) + \
                 (self.sigma_1 * self.tau * gk / self.h_dash(i)) + \
                 ((1 - self.sigma_1) * self.tau * gj / self.h_dash(i))

        return f1_k_i

    def get_f2(self, i, k, j):
        f2_k_i = self.x_k.get_i_j(i, k) - self.x.get_i_j(i, j) - \
                 0.5 * self.tau * (self.v_k.get_i_j(i, k) + self.v.get_i_j(i, j))
        return f2_k_i

    def get_f3(self, i, k, j):
        f3_k_i = ((self.x_k.get_i_j(i + 1, k) - self.x_k.get_i_j(i, k)) / self.h_dash(i + 1)) - (
                1 / self.ρ_k.get_i_j(i, k))
        return f3_k_i

    def get_f4(self, i, k, j):
        gl = self.p_k.get_i_j(i, k) + self.func_Ω(self.v_k.get_i_j(i, k),
                                                  self.v_k.get_i_j(i + 1, k),
                                                  self.ρ_k.get_i_j(i, k),
                                                  self.h_dash(i + 1)
                                                  )
        gj = self.p.get_i_j(i, j) + self.func_Ω(self.v.get_i_j(i, j),
                                                self.v.get_i_j(i + 1, j),
                                                self.ρ.get_i_j(i, j),
                                                self.h_dash(i + 1)
                                                )

        vs05 = 0.5 * (self.v_k.get_i_j(i + 1, k) - self.v_k.get_i_j(i, k)) / self.h_dash(i) + \
               0.5 * (self.v.get_i_j(i + 1, j) - self.v.get_i_j(i, j)) / self.h_dash(i)

        wk_s = (self.func_W(self.t_k.get_i_j(i + 1, k),
                            self.t_k.get_i_j(i, k),
                            self.h_dash(i + 1)) -
                self.func_W(self.t_k.get_i_j(i, k),
                            self.t_k.get_i_j(i - 1, k),
                            self.h_dash(i))) / self.h(i)

        wj_s = (self.func_W(self.t.get_i_j(i + 1, j),
                            self.t.get_i_j(i, j),
                            self.h_dash(i + 1)) -
                self.func_W(self.t.get_i_j(i, j),
                            self.t.get_i_j(i - 1, j),
                            self.h_dash(i))) / self.h(i)

        f4_k_i = (self.e_k.get_i_j(i, k) - self.e.get_i_j(i, j)) + \
                 self.tau * (self.sigma_1 * gl + (1 - self.sigma_1) * gj) * vs05 + \
                 self.tau * (self.sigma_2 * wk_s + (1 - self.sigma_2) * wj_s)

        return f4_k_i

    def get_f5(self, i, k, j):
        f5_l_i = self.func_W(self.t_k.get_i_j(i, k), self.t_k.get_i_j(i - 1, k), self.h_dash(i)) + \
                 self.func_K(self.t_k.get_i_j(i, k)) * (
                         self.t_k.get_i_j(i, k) - self.t_k.get_i_j(i - 1, k)) / self.h_dash(i)

        return f5_l_i

    def get_f6(self, i, k, j):
        f6_l_i = self.p_k.get_i_j(i, k) - self.func_P(self.ρ_k.get_i_j(i, k), self.t_k.get_i_j(i, k))
        return f6_l_i

    def get_f7(self, i, k, j):
        f7_l_i = self.e_k.get_i_j(i, k) - self.func_E(self.t_k.get_i_j(i, k), self.ρ_k.get_i_j(i, k))
        return f7_l_i

    def get_f8(self, i, k, j):
        # f8_l_i = self.K_l.get_i_j(i, l) - self.func_K(self.t_l.get_i_j(i, l))
        return 0.  # f8_l_i

    def get_f9(self, i, k, j):
        f9_l_i = self.ω_k.get_i_j(i, k) - self.func_Ω(self.v_k.get_i_j(i, k),
                                                      self.v_k.get_i_j(i + 1, k),
                                                      self.ρ_k.get_i_j(i, k),
                                                      self.h_dash(i)
                                                      )
        return f9_l_i

    def calculate_layer_j(self, j):
        # self.max_k_iters
        # будем использовать такой же меш только вместо измерения по времени будет k
        self.A_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='A_k')
        self.B_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='B_k')
        self.C_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='C_k')
        self.F_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='F_k')

        self.D_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='D_k')

        self.G_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='G_k')

        self.E_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='E_k')

        self.F2_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='F2_k')

        self.Y_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='Y_k')
        self.ρ_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='ρ_k')

        self.v_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='v_k')

        self.x_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='x_k')

        self.p_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='p_k')

        self.e_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='e_k')

        self.t_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='t_k')

        self.g_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='g_k')
        self.ω_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='ω_k')

        self.W_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='W_k')

        self.K_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t, name='K_k')

        self.delta_v_k = Mesh(N=self.N, T=self.max_k_iters, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t,
                              name='delta_v_k')

        # #для решения второго уравнения относительно температуры
        # self.delta_t_l= Mesh(N=self.N, T=_n, tau=self.tau, h=self.h, a_s=self.a_s, a_t=self.a_t,
        #                       name='delta_t_l', is_half=False)

        # заполняем по формулам (3,8) со страницы 209

        # заполним начальные условия для k = 0
        # x,v,p,ρ,t,e,W,g,K,ω
        for i in range(-1, self.v.N + 2):
            # начальные значения для плотности
            self.ρ_k.set_i_j(i, 0, self.ρ.get_i_j(i, j))
            self.v_k.set_i_j(i, 0, self.v.get_i_j(i, j))
            self.x_k.set_i_j(i, 0, self.x.get_i_j(i, j))

            self.p_k.set_i_j(i, 0, self.p.get_i_j(i, j))
            self.e_k.set_i_j(i, 0, self.e.get_i_j(i, j))
            self.t_k.set_i_j(i, 0, self.t.get_i_j(i, j))
            self.ω_k.set_i_j(i, 0, self.ω.get_i_j(i, j))
            self.W_k.set_i_j(i, 0, self.W.get_i_j(i, j))
            self.K_k.set_i_j(i, 0, self.K.get_i_j(i, j))

        for k in range(0, self.max_k_iters):

            for i in range(1, self.v.N):
                მΩ_by_მv = self.func_მΩ_by_მv(self.v_k.get_i_j(i - 1, k),
                                              self.v_k.get_i_j(i, k),
                                              self.ρ_k.get_i_j(i - 1, k),
                                              self.h(i - 1))

                მg_by_მρ = self.func_მg_by_მρ(self.v_k.get_i_j(i - 1, k),
                                              self.v_k.get_i_j(i, k),
                                              self.ρ_k.get_i_j(i - 1, k),
                                              self.t_k.get_i_j(i - 1, k),
                                              self.h(i - 1)
                                              )
                ai_prev = self.sigma_1 * self.tau * (
                        (0.5 * self.tau / self.h(i - 1)) * (self.ρ_k.get_i_j(i - 1, k) ** 2) * მg_by_მρ + მΩ_by_მv)
                A_k_i = ai_prev / self.h_dash(i)
                self.A_k.set_i_j(i, k, A_k_i)

            for i in range(1, self.N):
                bi = self.sigma_1 * self.tau * (
                        (0.5 * self.tau / self.h(i)) * (self.ρ_k.get_i_j(i, k) ** 2) * self.func_მg_by_მρ(
                    self.v_k.get_i_j(i, k),
                    self.v_k.get_i_j(i + 1, k),
                    self.ρ_k.get_i_j(i, k),
                    self.t_k.get_i_j(i, k),
                    self.h_dash(i)
                ) +
                        self.func_მΩ_by_მv(self.v_k.get_i_j(i, k),
                                           self.v_k.get_i_j(i + 1, k),
                                           self.ρ_k.get_i_j(i, k),
                                           self.h_dash(i)))
                B_k_i = bi / self.h_dash(i)
                self.B_k.set_i_j(i, k, B_k_i)
            self.B_k.set_i_j(0, k, 0)  # стр 229 alpha1 = B0/C0

            for i in range(1, self.N):
                ai = self.sigma_1 * self.tau * (
                        (0.5 * self.tau / self.h(i)) * (self.ρ_k.get_i_j(i, k) ** 2) * self.func_მg_by_მρ(
                    self.v_k.get_i_j(i, k),
                    self.v_k.get_i_j(i + 1, k),
                    self.ρ_k.get_i_j(i, k),
                    self.t_k.get_i_j(i, k),
                    self.h_dash(i)
                ) +
                        self.func_მΩ_by_მv(self.v_k.get_i_j(i, k),
                                           self.v_k.get_i_j(i + 1, k),
                                           self.ρ_k.get_i_j(i, k),
                                           self.h_dash(i)))
                bi_prev = self.sigma_1 * self.tau * (
                        (0.5 * self.tau / self.h(i - 1)) * (self.ρ_k.get_i_j(i - 1, k) ** 2) * self.func_მg_by_მρ(
                    self.v_k.get_i_j(i - 1, k),
                    self.v_k.get_i_j(i, k),
                    self.ρ_k.get_i_j(i - 1, k),
                    self.t_k.get_i_j(i - 1, k),
                    self.h(i - 1)
                ) +
                        self.func_მΩ_by_მv(self.v_k.get_i_j(i - 1, k),
                                           self.v_k.get_i_j(i, k),
                                           self.ρ_k.get_i_j(i - 1, k),
                                           self.h(i - 1)))
                C_k_i = 1 + (ai + bi_prev) / self.h_dash(i)
                self.C_k.set_i_j(i, k, C_k_i)
            self.C_k.set_i_j(0, k, 1)  # стр 229 alpha1 = B0/C0

            for i in range(1, self.v.N):
                # fixme убрать в отдельную функцию
                Y_k_i = (self.ρ_k.get_i_j(i, k) ** 2) * (self.get_f3(i, k, j) -
                                                         (self.get_f2(i + 1, k, j) - self.get_f2(i, k,
                                                                                                 j)) / self.h_dash(i))

                Y_k_i_prev = (self.ρ_k.get_i_j(i - 1, k) ** 2) * (self.get_f3(i - 1, k, j) -
                                                                  (self.get_f2(i, k, j) - self.get_f2(i - 1, k,
                                                                                                      j)) / self.h_dash(
                            i))

                q1 = (self.func_მg_by_მρ(self.v_k.get_i_j(i, k),
                                         self.v_k.get_i_j(i + 1, k),
                                         self.ρ_k.get_i_j(i, k),
                                         self.t_k.get_i_j(i, k),
                                         self.h_dash(i)
                                         ) * Y_k_i +
                      self.get_f6(i, k, j) + self.get_f9(i, k, j))

                q2 = (self.func_მg_by_მρ(self.v_k.get_i_j(i - 1, k),
                                         self.v_k.get_i_j(i, k),
                                         self.ρ_k.get_i_j(i - 1, k),
                                         self.t_k.get_i_j(i, k),
                                         self.h(i - 1)
                                         ) * Y_k_i_prev +
                      self.get_f6(i - 1, k, j) + self.get_f9(i - 1, k, j))

                F_k_i = -self.get_f1(i, k, j) + self.sigma_1 * self.tau * (q1 - q2) / self.h_dash(i)
                self.F_k.set_i_j(i, k, F_k_i)

            self.F_k.set_i_j(0, k, 0)  # стр 229 alpha1 = F0/C0
            self.F_k.set_i_j(self.N, k, 0)  # страница 229 начальные условия

            progonka = Progonka(self.N,
                                C0=self.C_k.get_i_j(0, k),
                                B0=self.B_k.get_i_j(0, k),
                                F0=self.F_k.get_i_j(0, k),
                                AN=self.A_k.get_i_j(self.N, k),
                                CN=self.C_k.get_i_j(self.N, k),
                                FN=self.F_k.get_i_j(self.N, k),
                                YN=0)  # начальное условие

            for i in range(1, self.N):
                progonka.set_A_i(i, self.A_k.get_i_j(i, k))
                progonka.set_B_i(i, self.B_k.get_i_j(i, k))
                progonka.set_C_i(i, self.C_k.get_i_j(i, k))
                progonka.set_F_i(i, self.F_k.get_i_j(i, k))

            progonka.check_solution(print_results=False)
            # assert progonka.check()
            # ответы на первой итерации
            _delta_v_k = progonka.run()  # (решение для уравнения 3,7)

            for i in range(0, self.N + 1):
                self.delta_v_k.set_i_j(i, k + 1, _delta_v_k[i])

            # найдем v на итерации (k+1)
            for i in range(0, self.N + 2):
                self.v_k.set_i_j(i, k + 1, (self.v_k.get_i_j(i, k) + self.delta_v_k.get_i_j(i, k + 1)))

            for i in range(0, self.N + 2):
                delta_x_k_i = 0.5 * self.tau * self.delta_v_k.get_i_j(i, k + 1) - self.get_f2(i, k, j)
                self.x_k.set_i_j(i, k + 1, self.x_k.get_i_j(i, k) + delta_x_k_i)

            for i in range(0, self.N + 1):
                _delta_x_k_i = self.x_k.get_i_j(i, k + 1) - self.x_k.get_i_j(i, k)
                _delta_x_k_i_plus = self.x_k.get_i_j(i + 1, k + 1) - self.x_k.get_i_j(i + 1, k)
                delta_xs = (_delta_x_k_i_plus - _delta_x_k_i) / self.h_dash(i + 1)
                f3 = self.get_f3(i, k, j)
                delta_ρ_k_i = (-f3 - delta_xs) * (self.ρ_k.get_i_j(i, k) ** 2)
                self.ρ_k.set_i_j(i, k + 1, self.ρ_k.get_i_j(i, k) + delta_ρ_k_i)

            self.ρ_k.set_i_j(-1, k + 1, self.ρ_k.get_i_j(0, k + 1))
            self.ρ_k.set_i_j(self.N + 1, k + 1, self.ρ_k.get_i_j(self.N, k + 1))

            for i in range(0, self.v.N + 1):
                vs05 = 0.5 * (self.v_k.get_i_j(i + 1, k) - self.v_k.get_i_j(i, k)) / self.h_dash(i) + \
                       0.5 * (self.v.get_i_j(i + 1, j) - self.v.get_i_j(i, j)) / self.h_dash(i)

                E_k_i = self.func_მE_by_მT() + self.tau * self.sigma_1 * vs05 * self.func_მP_by_მT(
                    self.ρ_k.get_i_j(i, k))
                self.E_k.set_i_j(i, k, E_k_i)

            for i in range(0, self.v.N + 1):
                vs05 = 0.5 * (self.v_k.get_i_j(i + 1, k) - self.v_k.get_i_j(i, k)) / self.h_dash(i) + \
                       0.5 * (self.v.get_i_j(i + 1, j) - self.v.get_i_j(i, j)) / self.h_dash(i)

                f58 = self.get_f5(i, k, j) - (self.t_k.get_i_j(i, k) - self.t_k.get_i_j(i - 1, k)) / self.h_dash(
                    i) * self.get_f8(i, k, j)

                f58_next_x1 = self.get_f5(i + 1, k, j)
                f58_next_x2 = ((self.t_k.get_i_j(i + 1, k) - self.t_k.get_i_j(i, k)) / self.h_dash(i + 1))
                f58_next_x3 = self.get_f8(i + 1, k, j)
                f58_next = f58_next_x1 - f58_next_x2 * f58_next_x3

                F2_k_i = -self.get_f4(i, k, j) + \
                         self.tau * self.sigma_1 * vs05 * self.get_f6(i, k, j) + \
                         self.get_f7(i, k, j) + \
                         self.tau * self.sigma_2 * (f58_next - f58) / self.h_dash(i)

                self.F2_k.set_i_j(i, k, F2_k_i)

            for i in range(0, self.v.N + 1):
                delta_t_k = self.F2_k.get_i_j(i, k) / self.E_k.get_i_j(i, k)
                self.t_k.set_i_j(i, k + 1, (self.t_k.get_i_j(i, k) + delta_t_k))

            for i in range(0, self.N + 1):
                delta_ρ_k_i = self.ρ_k.get_i_j(i, k + 1) - self.ρ_k.get_i_j(i, k)
                delta_t_k_i = self.t_k.get_i_j(i, k + 1) - self.t_k.get_i_j(i, k)

                delta_p_k_i = -self.get_f6(i, k, j) + \
                              (self.func_მP_by_მρ(self.t_k.get_i_j(i, k)) * delta_ρ_k_i +
                               self.func_მP_by_მT(self.ρ_k.get_i_j(i, k) * delta_t_k_i))

                self.p_k.set_i_j(i, k + 1, self.p_k.get_i_j(i, k) + delta_p_k_i)

            for i in range(0, self.N + 1):
                delta_ρ_k_i = self.ρ_k.get_i_j(i, k + 1) - self.ρ_k.get_i_j(i, k)
                delta_v_k_i = self.v_k.get_i_j(i, k + 1) - self.v_k.get_i_j(i, k)
                delta_v_k_i_next = self.v_k.get_i_j(i + 1, k + 1) - self.v_k.get_i_j(i + 1, k)

                delta_ω_k_i = -self.get_f9(i, k, j) + \
                              self.func_მΩ_by_მρ(self.v_k.get_i_j(i, k),
                                                 self.v_k.get_i_j(i + 1, k),
                                                 self.ρ_k.get_i_j(i, k),
                                                 self.h_dash(i)) * delta_ρ_k_i + \
                              self.func_მΩ_by_მv(self.v_k.get_i_j(i, k),
                                                 self.v_k.get_i_j(i + 1, k),
                                                 self.ρ_k.get_i_j(i, k),
                                                 self.h_dash(i)) * delta_v_k_i + \
                              -self.func_მΩ_by_მv(self.v_k.get_i_j(i, k),
                                                  self.v_k.get_i_j(i + 1, k),
                                                  self.ρ_k.get_i_j(i, k),
                                                  self.h_dash(i)) * delta_v_k_i_next

                self.ω_k.set_i_j(i, k + 1, self.ω_k.get_i_j(i, k) + delta_ω_k_i)

            for i in range(0, self.N + 1):
                delta_p_k_i = self.p_k.get_i_j(i, k + 1) - self.p_k.get_i_j(i, k)
                delta_ω_k_i = self.ω_k.get_i_j(i, k + 1) - self.ω_k.get_i_j(i, k)
                delta_g_k_i = delta_p_k_i + delta_ω_k_i
                self.g_k.set_i_j(i, k + 1, self.g_k.get_i_j(i, k) + delta_g_k_i)

            for i in range(0, self.N + 1):
                delta_ρ_k_i = self.ρ_k.get_i_j(i, k + 1) - self.ρ_k.get_i_j(i, k)
                delta_ρ_k_i_prev = self.ρ_k.get_i_j(i - 1, k + 1) - self.ρ_k.get_i_j(i - 1, k)
                delta_t_k_i = self.t_k.get_i_j(i, k + 1) - self.t_k.get_i_j(i, k)
                delta_t_k_i_prev = self.t_k.get_i_j(i - 1, k + 1) - self.t_k.get_i_j(i - 1, k)

                delta_K_k_i = -self.get_f8(i, k, j) + \
                              self.func_მK_by_მρ() * delta_ρ_k_i + \
                              self.func_მK_by_მρ_prev() * delta_ρ_k_i_prev + \
                              self.func_მK_by_მT(self.t_k.get_i_j(i, k)) * delta_t_k_i + \
                              self.func_მK_by_მT_prev() * delta_t_k_i_prev

                self.K_k.set_i_j(i, k + 1, self.K_k.get_i_j(i, k) + delta_K_k_i)

            for i in range(0, self.N + 1):
                delta_K_k_i = self.K_k.get_i_j(i, k + 1) - self.K_k.get_i_j(i, k)

                delta_t_k_i = self.t_k.get_i_j(i, k + 1) - self.t_k.get_i_j(i, k)
                delta_t_k_i_prev = self.t_k.get_i_j(i - 1, k + 1) - self.t_k.get_i_j(i - 1, k)

                delta_W_k_i = -self.get_f5(i, k, j) - \
                              (self.t_k.get_i_j(i, k) - self.t_k.get_i_j(i - 1, k)) / self.h_dash(i) * delta_K_k_i - \
                              self.K_k.get_i_j(i, k) * (delta_t_k_i - delta_t_k_i_prev) / self.h_dash(i)

                self.W_k.set_i_j(i, k + 1, self.W_k.get_i_j(i, k) + delta_W_k_i)

            for i in range(0, self.N + 1):
                delta_ρ_k_i = self.ρ_k.get_i_j(i, k + 1) - self.ρ_k.get_i_j(i, k)
                delta_t_k_i = self.t_k.get_i_j(i, k + 1) - self.t_k.get_i_j(i, k)
                delta_e_k_i = -self.get_f7(i, k, j) + \
                              self.func_მE_by_მρ() * delta_ρ_k_i + \
                              self.func_მE_by_მT() * delta_t_k_i

                self.e_k.set_i_j(i, k + 1, self.e_k.get_i_j(i, k) + delta_e_k_i)

        for i in range(-1, self.N + 2):
            self.v.set_i_j(i, j + 1, self.v_k.get_i_j(i, self.max_k_iters))
            self.x.set_i_j(i, j + 1, self.x_k.get_i_j(i, self.max_k_iters))
            self.ρ.set_i_j(i, j + 1, self.ρ_k.get_i_j(i, self.max_k_iters))

            self.t.set_i_j(i, j + 1, self.t_k.get_i_j(i, self.max_k_iters))
            self.p.set_i_j(i, j + 1, self.p_k.get_i_j(i, self.max_k_iters))
            self.e.set_i_j(i, j + 1, self.e_k.get_i_j(i, self.max_k_iters))
            self.W.set_i_j(i, j + 1, self.W_k.get_i_j(i, self.max_k_iters))
            self.g.set_i_j(i, j + 1, self.g_k.get_i_j(i, self.max_k_iters))
            self.K.set_i_j(i, j + 1, self.K_k.get_i_j(i, self.max_k_iters))
            self.ω.set_i_j(i, j + 1, self.ω_k.get_i_j(i, self.max_k_iters))


def experiment_1():
    """
    Зависимость результата от шага сетки
    :return:
    """
    adiabat1 = AdiabatWave(N=50, T=100, tau=0.01, H=0.2, a_s=0, a_t=0, max_k_iters=10, sigma=0.5, nu=0.1,
                           experiment_prefix='experiment_1_1', save_plots=True, save_params=True, show_plots=False,
                           ρ_L=1, ρ_R=1, V_L=1, V_R=0, P_L=0, P_R=0, x_right=6, x_left=4
                           )
    adiabat1.run()

    adiabat2 = AdiabatWave(N=100, T=100, tau=0.01, H=0.1, a_s=0, a_t=0, max_k_iters=10, sigma=0.5, nu=0.1,
                           experiment_prefix='experiment_1_2', save_plots=True, save_params=True, show_plots=False,
                           ρ_L=1, ρ_R=1, V_L=1, V_R=0, P_L=0, P_R=0, x_right=6, x_left=4
                           )
    adiabat2.run()

    adiabat3 = AdiabatWave(N=200, T=100, tau=0.01, H=0.05, a_s=0, a_t=0, max_k_iters=10, sigma=0.5, nu=0.1,
                           experiment_prefix='experiment_1_3', save_plots=True, save_params=True, show_plots=False,
                           ρ_L=1, ρ_R=1, V_L=1, V_R=0, P_L=0, P_R=0, x_right=6, x_left=4
                           )
    adiabat3.run()


if __name__ == "__main__":
    experiment_1()
