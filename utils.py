import numpy as np

def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    grad = np.zeros_like(w)
    f0 = function(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_p[i] += eps
        fp = function(w_p)
        grad[i] = (fp - f0) / eps
    return grad
