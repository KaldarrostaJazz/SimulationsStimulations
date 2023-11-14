class RungeKutta4:
    def __init__(self, fun, t0, y0, t_bound, step_size):
        if (len(y0) != len(fun(t0,y0))):
            print('Initial state not compatible with the system')
        else:
            self.fun = fun
            self.t0 = t0
            self.y0 = y0
            self.t_bound = t_bound
            self.step_size = step_size
