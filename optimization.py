import time
import numpy as np
from scipy import sparse as sp
from scipy.special import expit
from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=0.1, step_beta=0.0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """

        self.loss = loss_function
        self.alpha = float(step_alpha)
        self.beta = float(step_beta)
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.args = kwargs

        # тут будут веса
        self.w = None

        if self.loss == "binary_logistic":
            self.oracle = BinaryLogistic(**self.args)
        else:
            raise ValueError("Unknown loss function")

    def fit(self, X, y, w_0=None, trace=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
            """
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0.copy()

        f_prev = self.oracle.func(X, y, self.w)

        if trace:
            history = {'time': [0.0], 'func': [f_prev], 'accuracy': []}
            t_prev = time.perf_counter()
            total_time = 0
        else:
            history = {}
        for k in range(self.max_iter):
            eta = self.alpha / ((k + 1) ** self.beta)
            grad = self.oracle.grad(X, y, self.w)
            self.w -= eta * grad
            f_curr = self.oracle.func(X, y, self.w)

            if trace:
                t_now = time.perf_counter()
                total_time += t_now - t_prev          # копим общее время
                history['time'].append(total_time)    # логируем накопленное
                history['func'].append(f_curr)
                t_prev = t_now

            if abs(f_curr - f_prev) < self.tolerance:
                break
            f_prev = f_curr

            if trace:
                acc = None
                if X_test is not None and y_test is not None:
                    acc = (self.predict(X_test) == y_test).mean()
                history['accuracy'].append(acc)

        return history if trace else None

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        res = X.dot(self.w) if sp.issparse(X) else X @ self.w
        p = expit(res)

        return (p >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        res = X.dot(self.w) if sp.issparse(X) else X @ self.w
        p1 = expit(res)
        p0 = 1 - p1

        return np.vstack([p0, p1]).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1.0, step_beta=0.0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(
            loss_function=loss_function,
            step_alpha=step_alpha,
            step_beta=step_beta,
            tolerance=tolerance,
            max_iter=max_iter,
            **kwargs
        )
        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        rng = np.random.seed(self.random_seed)
        self.w = np.zeros(X.shape[1]) if w_0 is None else w_0.copy()

        n_samples = X.shape[0]
        history = {}

        if trace:
            history = {
                'epoch_num': [0.0],
                'time': [0.0],
                'func': [self.oracle.func(X, y, self.w)],
                'weights_diff': [0.0],
                'accuracy': [0.0]
            }
            t_prev = time.perf_counter()
            total_time = 0

        processed_objects = 0
        prev_epoch = 0.0
        grad = np.zeros_like(self.w)

        for epoch in range(self.max_iter):
            ids = np.arange(n_samples)
            np.random.shuffle(ids)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_ids = ids[start: end]
                X_batch = X[batch_ids]
                y_batch = y[batch_ids]

                eta = self.alpha / ((processed_objects / self.batch_size + 1) ** self.beta)

                grad = self.oracle.grad(X_batch, y_batch, self.w)
                w_old = self.w.copy()
                self.w -= eta * grad
                processed_objects += self.batch_size
                curr_epoch = processed_objects / n_samples
                if trace and (curr_epoch - prev_epoch >= log_freq):
                    t_now = time.perf_counter()
                    elapsed = t_now - t_prev
                    total_time += elapsed

                    f_val = self.oracle.func(X, y, self.w)
                    weights_diff = float(np.linalg.norm(self.w - w_old) ** 2)

                    history['epoch_num'].append(curr_epoch)
                    history['time'].append(total_time)  # накопленное время
                    history['func'].append(f_val)
                    history['weights_diff'].append(weights_diff)

                    t_prev = t_now

                    if X_test is not None and y_test is not None:
                        acc = (self.predict(X_test) == y_test).mean()
                    else:
                        acc = None
                    history['accuracy'].append(acc)
                    prev_epoch = curr_epoch
            if np.linalg.norm(grad) < self.tolerance:
                break
        return history if trace else None
