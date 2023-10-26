def runge_kutta(f, x0, y0, h, n):
    """
    Implements the fourth-order Runge-Kutta method for solving ODEs.

    Parameters:
    f : function
        The function f(x, y) in the ODE dy/dx = f(x, y).
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y at x0.
    h : float
        Step size.
    n : int
        Number of iterations/number of steps.

    Returns:
    List of approximated values of y at each step.
    """
    y = [y0]
    x = x0
    for i in range(n):
        k1 = h * f(x, y[-1])
        k2 = h * f(x + 0.5 * h, y[-1] + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y[-1] + 0.5 * k2)
        k4 = h * f(x + h, y[-1] + k3)
        y.append(y[-1] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
        x += h
    return y
