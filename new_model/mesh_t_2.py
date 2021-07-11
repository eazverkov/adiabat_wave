import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Mesh(object):
    def __init__(self, N, T, tau, h, a_s=0, a_t=0, name='mesh example'):
        """
        Класс для работы с сеткой (страница 98)
        с целым шагом по времени
        U - функция определенная на сетке

        :param N:  число узлов по s
        :param T: число узлов по t
        :param tau:
        :param h:
        :param a_s: начальная точка по иксу
        :param a_t: начальная точка по t
        :param name:  mesh name
        :param is_half: определена функция в целых или полуцелых точках
        is_half = False для x и V
        is_half = True для дваления энергии и плотности
        """
        self.name = name
        self.N = N
        self.T = T

        self.h = h  # шаг по иксу
        self.tau = tau  # шаг по игреку

        # реальные размеры сетки чтобы можно было брать 1/2
        self.a_s = a_s
        self.a_t = a_t

        # меш лежащий внутри и используемый для в технических целях (содержит в себе целые и полуцелые точки,
        # в том числе и со служебными -1)
        self._mesh = np.zeros((self._count_of_t_cells, self._count_of_s_cells))

    def __str__(self):
        return f"Mesh  tau: {self.tau}  h:{self.h}  N: {self.N}, T: {self.T}  "

    def __repr__(self):
        return f"Mesh  tau: {self.tau}  h:{self.h}  N: {self.N}, T: {self.T} "

    @property
    def _count_of_s_cells(self):
        """
        !служебная функция
        количество полуячеек по s которые должны лежать в _mesh с учетом дополнительных слева и справа
        :return:
        """
        return self.N + 3

    @property
    def _count_of_t_cells(self):
        """
        !служебная функция
        количество ячеек по t  которые должны лежать в _mesh
        :return:
        """
        return self.T + 1

    def get_i_j(self, i, j):
        """
        :param i:  Ячейка по оси s
        :param j:  Ячейка по оси t
        :return:
        """
        _i = int(i + 1)
        _j = int(j)
        return self._mesh[_j, _i]

    def set_i_j(self, i, j, value):
        """
        :param i:  Ячейка по оси s
        :param j:  Ячейка по оси t
        :param value:  записываемое значение
        :return:
        """
        _i = int(i + 1)
        _j = int(j)
        self._mesh[_j, _i] = value

    def fill_mesh_with_test_s_values(self):
        for j in np.linspace(0, self.T, self._count_of_t_cells, endpoint=True):
            for i in np.linspace(-1, self.N + 1, self._count_of_s_cells, endpoint=True):
                _i = int(i + 1)
                _j = int(j)
                self._mesh[_j, _i] = i + j

    def _print_all_mesh(self):
        """
        Функция для печати всей служебной сетки в удобной форме для дебаггинга
        """
        print(self._mesh[::-1, :])

    # fixme проверить как печатается
    def _print_pretty_mesh(self):
        """
        Функция для печати служебной сетки в удобной форме для дебаггинга
        только целые или полуцелые ячейки
        без крайних слева и справа
        """
        _i = [int(i + 1) for i in range(0, self.N + 1)]
        print(self._mesh[::-1, _i])

    def plot_mesh(self, additional_title=' ', save_path=None, show_plot=False):
        _i = [int(i + 1) for i in range(0, self.N + 1)]
        plt.figure(figsize=(5, 5))
        plt.title(f'{self.name} heatmap' + additional_title)
        sns.heatmap(self._mesh[::-1, _i], cmap="coolwarm")
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()

    def good_points(self):
        _i = [int(i + 1) for i in range(0, self.N + 1)]
        return _i

    def left_derivative_by_s(self, i, j):
        return (self.get_i_j(i, j) - self.get_i_j(i - 1, j)) / self.h

    def right_derivative_by_s(self, i, j):
        return (self.get_i_j(i + 1, j) - self.get_i_j(i, j)) / self.h

    def central_derivative_by_s(self, i, j):
        return (self.get_i_j(i + 1, j) - self.get_i_j(i - 1, j)) / (2 * self.h)

    def derivative_by_t(self, i, j):
        return (self.get_i_j(i, j + 1) - self.get_i_j(i, j)) / self.tau

    def sigma_approximation(self, i, j, sigma):
        return sigma * self.get_i_j(i, j + 1) + (1 - sigma) * self.get_i_j(i, j)

    def second_derivative_by_s(self, i, j):
        return (self.get_i_j(i + 1, j) - 2 * self.get_i_j(i, j) + self.get_i_j(i - 1, j)) / (self.h ** 2)
