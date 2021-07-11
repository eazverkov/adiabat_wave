"""
учебник страница 199

реализация метода прогонки для системы
A(i)y(i-1) - C(i)y(i) + B(i)y(i+1) = -F(i)
____________________________
| A1 | -C1 | B1 | 0 | 0 | 0
|_______________________
|  0 | A2 | -C2 | B2 | 0 | 0 |
|_____________________________
| . ..............

*

(y0, y1, ...yN)


=( -F1, -F2 ... -FN)

"""
import numpy as np


class Progonka:
    def __init__(self, N, C0, B0, F0, AN, CN, FN, YN=None):
        """
        :param M:  матрица из коэффициентов A, B, C
        :param F:

        F.shape

        """
        self.N = N

        self.n = N + 1  # self.M.shape[1]  # количество переменных

        self.M = np.zeros((self.n - 2, self.n))
        self.F = np.zeros((self.n - 2,))

        self.C0 = C0
        self.B0 = B0
        self.F0 = F0
        self.AN = AN
        self.CN = CN
        self.FN = FN

        self.YN = YN

        assert self.C0 != 0
        if not self.YN is None:
            self.χ_2 = 0
            self.ν_2 = 0
        else:
            assert self.CN != 0
            self.χ_2 = AN / CN
            self.ν_2 = FN / CN

        self.χ_1 = B0 / C0
        self.ν_1 = F0 / C0

        self.alpha = np.zeros((self.n,))
        self.beta = np.zeros((self.n,))

        self.y = np.zeros((self.n,))

    # todo
    def check(self):
        """
        функция для проверку условий
        :return:
        """
        for i in range(1, self.N):
            if self.get_A_i(i) <= 0:
                print(f'self.get_A_i(i) <= 0  i={i}')
                # raise
                return False

            if self.get_B_i(i) <= 0:
                print(f'self.get_B_i(i) <= 0  i={i}')
                # raise
                return False

            if self.get_C_i(i) < (self.get_A_i(i) + self.get_B_i(i)):
                print('self.get_C_i(i) < (self.get_A_i(i) + self.get_B_i(i))')
                return False

        if (self.χ_1) < 0 or (self.χ_1 > 1):
            print('(self.χ_1) < 0 or (self.χ_1 > 1)')
            return False

        if (self.χ_2) < 0 or (self.χ_2 > 1):
            print('(self.χ_2) < 0 or (self.χ_2 > 1)')
            return False

        if ((self.χ_1 + self.χ_2) >= 2):
            print('((self.χ_1 + self.χ_2) >= 2)')
            return False

        return True

    def get_A_i(self, i):
        if 1 <= i <= (self.n - 1):
            return self.M[i - 1, i - 1]
        else:
            return np.nan

    def set_A_i(self, i, A_i):
        if 1 <= i <= (self.n - 1):
            self.M[i - 1, i - 1] = A_i

    def get_B_i(self, i):
        if 1 <= i <= (self.n - 1):
            return self.M[i - 1, i + 1]
        else:
            return np.nan

    def set_B_i(self, i, B_i):
        if 1 <= i <= (self.n - 1):
            self.M[i - 1, i + 1] = B_i

    def get_C_i(self, i):
        if 1 <= i <= (self.n - 1):
            return self.M[i - 1, i]
        else:
            return np.nan

    def set_C_i(self, i, C_i):
        if 1 <= i <= (self.n - 1):
            self.M[i - 1, i] = C_i

    def get_F_i(self, i):
        return self.F[i - 1]

    def set_F_i(self, i, F_i):
        self.F[i - 1] = F_i

    def set_alpha_i(self, i, value):
        self.alpha[i - 1] = value

    def get_alpha_i(self, i):
        return self.alpha[i - 1]

    def set_beta_i(self, i, value):
        self.beta[i - 1] = value

    def get_beta_i(self, i):
        return self.beta[i - 1]

    def set_y_i(self, i, value):
        self.y[i] = value

    def get_y_i(self, i):
        return self.y[i]

    def run(self):

        self.set_alpha_i(1, self.χ_1)
        self.set_beta_i(1, self.ν_1)


        for i in range(1, self.N):
            self.set_alpha_i(i + 1, self.get_B_i(i) / (self.get_C_i(i) - self.get_A_i(i) * self.get_alpha_i(i)))
            self.set_beta_i(i + 1, (self.get_A_i(i) * self.get_beta_i(i) + self.get_F_i(i)) / (
                    self.get_C_i(i) - self.get_A_i(i) * self.get_alpha_i(i)))


        if self.YN is None:
            y_N = (self.ν_2 + self.χ_2 * self.get_beta_i(self.N)) / (1 - self.χ_2 * self.get_alpha_i(self.N))
        else:
            y_N = self.YN
        self.set_y_i(self.N, y_N)

        for i in range(self.N - 1, -1, -1):
            y_i = self.get_alpha_i(i + 1) * self.get_y_i(i + 1) + self.get_beta_i(i + 1)
            self.set_y_i(i, y_i)



        return self.y

    # todo обавить тест на граничных условиях
    # todo добавить асертов есть больше эпсилона разница
    def check_solution(self, print_results=False):
        delta = -self.C0 * self.get_y_i(0) + self.B0 * self.get_y_i(1) + self.F0
        if print_results:
            print('check_solution:', 0, delta)

        for i in range(1, self.N):
            delta = self.get_A_i(i) * self.get_y_i(i - 1) - self.get_C_i(i) * self.get_y_i(i) + self.get_B_i(
                i) * self.get_y_i(i + 1) + self.get_F_i(i)
            if print_results:
                print('check_solution:', i, delta)

        delta = self.AN * self.get_y_i(self.N - 1) - self.CN * self.get_y_i(self.N) + self.FN
        if print_results:
            print('check_solution:', self.N, delta)


def test_two():
    progonka = Progonka(N=4, C0=10, B0=1, F0=1, AN=1, CN=10, FN=1)

    A1 = 1
    A2 = 2
    A3 = 3

    progonka.set_A_i(1, A1)
    progonka.set_A_i(2, A2)
    progonka.set_A_i(3, A3)

    B1 = 4
    B2 = 5
    B3 = 6

    progonka.set_B_i(1, B1)
    progonka.set_B_i(2, B2)
    progonka.set_B_i(3, B3)

    C1 = 7
    C2 = 8
    C3 = 9

    progonka.set_C_i(1, C1)
    progonka.set_C_i(2, C2)
    progonka.set_C_i(3, C3)

    F1 = 1
    F2 = 2
    F3 = 3

    progonka.set_F_i(1, F1)
    progonka.set_F_i(2, F2)
    progonka.set_F_i(3, F3)

    assert progonka.get_A_i(1) == A1
    assert progonka.get_A_i(2) == A2
    assert progonka.get_A_i(3) == A3

    assert progonka.get_B_i(1) == B1
    assert progonka.get_B_i(2) == B2
    assert progonka.get_B_i(3) == B3

    assert progonka.get_C_i(1) == C1
    assert progonka.get_C_i(2) == C2
    assert progonka.get_C_i(3) == C3

    print(progonka.M)
    print(progonka.F)

    assert progonka.check()
    y = progonka.run()

    progonka.check_solution(print_results=True)


def test_three():
    progonka = Progonka(N=4, C0=10, B0=1, F0=1, AN=0, CN=10, FN=0, YN=0)

    A1 = 1
    A2 = 2
    A3 = 3

    progonka.set_A_i(1, A1)
    progonka.set_A_i(2, A2)
    progonka.set_A_i(3, A3)

    B1 = 4
    B2 = 5
    B3 = 6

    progonka.set_B_i(1, B1)
    progonka.set_B_i(2, B2)
    progonka.set_B_i(3, B3)

    C1 = 7
    C2 = 8
    C3 = 9

    progonka.set_C_i(1, C1)
    progonka.set_C_i(2, C2)
    progonka.set_C_i(3, C3)

    F1 = 1
    F2 = 2
    F3 = 3

    progonka.set_F_i(1, F1)
    progonka.set_F_i(2, F2)
    progonka.set_F_i(3, F3)

    assert progonka.get_A_i(1) == A1
    assert progonka.get_A_i(2) == A2
    assert progonka.get_A_i(3) == A3

    assert progonka.get_B_i(1) == B1
    assert progonka.get_B_i(2) == B2
    assert progonka.get_B_i(3) == B3

    assert progonka.get_C_i(1) == C1
    assert progonka.get_C_i(2) == C2
    assert progonka.get_C_i(3) == C3

    print(progonka.M)
    print(progonka.F)

    assert progonka.check()
    y = progonka.run()

    progonka.check_solution(print_results=True)


if __name__ == "__main__":
    test_two()
    print('_____')
    test_three()
