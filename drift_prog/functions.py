import numpy as np

def composition(f:callable, g:callable) -> callable:
    def cmp(x):
        return f(g(x))
    return cmp

def find_min_of(f: callable, range: tuple) -> float:
    left  = range[0]
    right = range[1]
    eps   = 0.0001
    while right - left > eps: 
        a = (left * 2 + right) / 3
        b = (left + right * 2) / 3
        if f(a) < f(b):
            right = b
        else:
            left = a
    return (left + right) / 2

def find_intersection(f: callable, g: callable, rng: tuple) -> float:
    l = rng[0]
    r = rng[1]
    if (f(l) < g(l)):
        return find_intersection(g, f, rng)
    else:
        if (f(r) > g(r)):
            raise NameError("functions don't intersect")
        m = l
        for _ in range(0, 20):
            m = (l + r) / 2
            if (f(m) - g(m) > 0):
                l = m
            else:
                r = m
        return m

def expection(eps: float, shift) -> float:
    return 1 / eps - shift

def exponential_expection(eps: float, shift: float) -> callable:
    def gamma_func(gamma: float) -> float:
        return eps * np.exp(-shift * gamma) / (1 + eps - np.exp(gamma))
    return gamma_func

def sub_gaussian(const: float) -> callable:
    def gamma_func(gamma: float) -> float:
        return np.exp(const / 2 * gamma**2)
    return gamma_func

def p(eps: float, shift: float) -> callable:
    exp2 = exponential_expection(eps, shift)
    return composition(lambda x: 1 / (1 - x), exp2)

def negative_drift_probability_(eps: float, shift: float) -> callable:
    p_actual = p(eps, shift)
    def func(dist: float, gamma: float) -> float:
        return dist / shift * p_actual(gamma) * np.exp(-gamma * dist)
    return func

def negative_drift_probability(eps: float, shift: float) -> callable:
    neg_pr = negative_drift_probability_(eps, shift)
    left   = np.log(shift * (1 + eps) / (1 + shift))
    right  = np.log(1 + eps) - 0.001
    exp2   = exponential_expection(eps, shift)

    middle = find_intersection(lambda _: 1, exp2, (left, right)) - 0.001

    def func(dist: float) -> float:
        gamma0 = find_min_of(lambda g: neg_pr(dist, g), (left, middle))
        return neg_pr(dist, gamma0)
    
    return func

def sub_gaussian_probability(eps: float, shift: float) -> callable:
    left  = np.log(shift * (1 + eps) / (1 + shift))
    right = np.log(1 + eps) - 0.001
    exp2 = exponential_expection(eps, shift)
    def func(dist: float, const: float) -> float:
        gauss = sub_gaussian(const)
        delta0 = find_intersection(exp2, gauss, (left, right))
        return np.exp(-dist / 2 * min(delta0, shift / const))
    return func

