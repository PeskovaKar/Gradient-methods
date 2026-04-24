import numpy as np
from scipy import sparse as sp
from scipy.special import expit


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.coef = l2_coef

    @staticmethod
    def make_dot(X, w):
        return X.dot(w) if sp.issparse(X) else X @ w

    @staticmethod
    def make_01(y):
        if np.array_equal(np.unique(y), np.array([-1, 1])):
            return (y + 1) / 2
        return y

    def l2_reg(self, w):
        if self.coef == 0:
            return 0, np.zeros_like(w)
        reg_val = 0.5 * self.coef * np.dot(w, w)
        reg_grad = self.coef * w
        return reg_val, reg_grad

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        y = self.make_01(y)
        z = self.make_dot(X, w)
        reg_val, _ = self.l2_reg(w)
        loss = np.mean(np.logaddexp(0.0, z) - y * z)

        return float(loss + reg_val)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        y = self.make_01(y)
        z = self.make_dot(X, w)
        p = expit(z)

        if sp.issparse(X):
            grad = X.T.dot(p - y)
            grad = np.asarray(grad).ravel()
        else:
            grad = X.T @ (p - y)

        _, reg_grad = self.l2_reg(w)
        grad = grad / X.shape[0] + reg_grad
        return grad
