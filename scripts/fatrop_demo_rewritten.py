print('\nRunning demo!\n')

import casadi as ca
from numpy import sin, cos, tan, pi

# Params
N = 20
L = 1

# Initial time
T0 = 10

# Initial pos/angle
p0 = [0, 0, 0]
pf = [0, 10, 0]

# Spherical obstacle 
p_obs = [0.2, 5]
r_obs = 1

# Control limits
vel_min = 0
vel_max = 1
delta_min = -ca.pi/6
delta_max = ca.pi/6

# symbolic states
x = ca.SX.sym('[px, py, theta]', 3, 1)
u = ca.SX.sym('[delta, vel]', 2, 1)
t = ca.SX.sym('t')

# EOMS
px, py, theta = x[0], x[1], x[2]
delta, vel = u[0], u[1]
px_dot = vel*ca.cos(theta)
py_dot = vel*ca.sin(theta)
theta_dot = vel/L*ca.tan(delta)

ode = ca.vertcat(px_dot, py_dot, theta_dot)
ode_func = ca.Function('ode_func', [x, u], [ode])

# Number of states and controls (3, 2)
nx = x.numel()
nu = u.numel()

# Decision variables
# New order [x, t, u]
V = ca.MX.sym('V', (nx+nu+1)*N + nx + 1)
# X = [V[(nx+nu+1)*i+1 : (nx+nu+1)*(i+1)-nu] for i in range(N+1)]
# U = [V[(nx+nu+1)*i+nx+1 : (nx+nu+1)*(i+1)] for i in range(N)]
# T = [V[(nx+nu+1)*i] for i in range(N+1)]

X = [V[(nx+nu+1)*i : (nx+nu+1)*(i+1)-nu-1] for i in range(N+1)]
U = [V[(nx+nu+1)*i+nx+1 : (nx+nu+1)*(i+1)] for i in range(N)]
T = [V[(nx+nu+1)*i+nx] for i in range(N+1)]


G = []
lbg = []
ubg = []
equality = []
lbx = []
ubx = []
x0 = []

# dt_sym = ca.SX.sym("dt_sym")
# sys = {}
# sys["x"] = x
# sys["u"] = u
# sys["p"] = dt_sym
# sys["ode"] = ode*dt_sym # Time scaling

# # integrator from 0 to 1 scaled by dt
# intg = ca.integrator('intg','rk',sys,0,1,{"simplify":True, "number_of_finite_elements": 4})

for i in range(N+1):
    # [T, px, py, ]
    if i == 0: # First state
        lbx += [p0[0], p0[1], p0[2], 0, delta_min, vel_min]
        ubx += [p0[0], p0[1], p0[2], ca.inf, delta_max, vel_max]
    elif i == N: # Last state
        lbx += [pf[0], pf[1], pf[2], 0]
        ubx += [pf[0], pf[1], pf[2], ca.inf]
    else: # In between states
        lbx += [-ca.inf, -ca.inf, -ca.inf, 0, delta_min, vel_min]
        ubx += [ca.inf, ca.inf, ca.inf, ca.inf, delta_max, vel_max]

    if i < N:
        # initial states
        x0 += [0, i*T0/N, ca.pi/2, T0, 0, 1] # Should probably fix px/py init to be a real lin interp

        # Do gap closing here
        dt = T[i]/N
        k1 = ode_func(X[i], U[i])
        k2 = ode_func(X[i] + dt/2 * k1, U[i])
        k3 = ode_func(X[i] + dt/2 * k2, U[i])
        k4 = ode_func(X[i] + dt * k3, U[i])
        x_next = X[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # x_next = intg(x0=X[i], u=U[i], p=dt)['xf']

        G.append(X[i+1]- x_next) # Gap closing on x ATTENTION: MUST BE IN FORM X[i+1] = F(X[i]) !!!
        G.append(T[i+1] - T[i]) # Gap closing on T
        lbg += (nx+1)*[0]
        ubg += (nx+1)*[0]
        equality += (nx+1)*[True]

        # obstacle avoidance
        pos = X[i][0:2]
        G.append(ca.sumsqr(pos-ca.vertcat(*p_obs)))
        lbg.append(r_obs**2)
        ubg.append(ca.inf)
        equality += [False]
    else:
        x0 += [0, i*T0/N, ca.pi/2, T0] # Should probably fix px/py init to be a real lin interp

opt_func = sum(T)

fatrop_opts = {
    'expand': True,
    'fatrop': {"mu_init": 0.1},
    'structure_detection': 'auto',
    'debug': True,
    'equality': equality
}

# (codegen of helper functions)
#options["jit"] = True
#options["jit_temp_suffix"] = False
#options["jit_options"] = {"flags": ["-O3"],"compiler": "ccache gcc"}

nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}

solver = ca.nlpsol('solver',"fatrop", nlp, fatrop_opts)

res = solver(x0 = x0,
    lbg = lbg,
    ubg = ubg,
    lbx = lbx,
    ubx = ubx,
)
