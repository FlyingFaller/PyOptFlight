print('\nRunning demo with casadi maps!\n')

import casadi as ca
import time
import subprocess

start_time = time.time()

# -------------------------------
# 1. Define the dynamics and integrator
# -------------------------------
# state (column vector) and control
x = ca.SX.sym('x', 3, 1)      # [px, py, theta]
u = ca.SX.sym('u', 2, 1)      # [delta, vel]

px, py, theta = x[0], x[1], x[2]
delta, vel = u[0], u[1]

L = 1  # bicycle length

# Bicycle model ODE (see LaValle’s Planning Algorithms)
px_dot    = vel * ca.cos(theta)
py_dot    = vel * ca.sin(theta)
theta_dot = vel/L * ca.tan(delta)

ode = ca.vertcat(px_dot, py_dot, theta_dot)

# The integrator uses a time–step “dt” (here, dt = T[k]/N)
dt = ca.SX.sym("dt")
int_dict = {'x': x, 'u': u, 'p': dt, 'ode': ode*dt}
intg = ca.integrator('intg', 'rk', int_dict, 0, 1,
                     {"simplify": True, "number_of_finite_elements": 4})

# Create a function that maps (x, u, dt) -> next state
# (We “reshape” the output so that F works on vectors instead of column–vectors.)
F = ca.Function('F', [x, u, dt], [intg(x0=x, u=u, p=dt)['xf']])

# -------------------------------
# 2. Set up decision variables (multiple–shooting)
# -------------------------------
N = 50           # number of intervals
T0_val = 10      # initial guess for time (all intervals are forced equal)

nx = 3
nu = 2
# Each stage (except the last) packs: [state (nx), time (1), control (nu)].
# The final stage has only [state, time].
V = ca.MX.sym('V', (N+1)*nx + (N+1)*1 + N*nu)

# For stage i = 0,...,N-1: extract state, time, control.
# For stage N: only state and time.
block = nx + nu + 1
X = [ V[ block*i : block*i + nx ] for i in range(N) ]
# Last stage (i = N): state only (followed by time)
X.append( V[ block*N : block*N + nx ] )
T  = [ V[ block*i + nx ] for i in range(N+1) ]
U  = [ V[ block*i + nx + 1 : block*i + nx + 1 + nu ] for i in range(N) ]

# -------------------------------
# 3. Build the initial guess and bounds
# -------------------------------
x0_guess = []
lbx = []
ubx = []
equality = []  # to mark equality constraints later

for k in range(N+1):
    # initial state guess: [0, k*T0/N, pi/2]
    x0_guess += [0, k*T0_val/N, ca.pi/2]
    if k == 0:
        lbx += [0, 0, 0]
        ubx += [0, 0, 0]
    elif k == N:
        lbx += [0, 10, -ca.inf]
        ubx += [0, 10, ca.inf]
    else:
        lbx += 3*[-ca.inf]
        ubx += 3*[ca.inf]
    # time variable guess and bounds
    x0_guess.append(T0_val)
    lbx.append(0)
    ubx.append(ca.inf)
    
    # For all but the last stage add control guess and bounds.
    if k < N:
        x0_guess += [0, 1]
        lbx += [-ca.pi/6, 0]
        ubx += [ ca.pi/6, 1]

# -------------------------------
# 4. Build constraints using maps and vectorized operations
# -------------------------------

g = []   # constraint expressions
lbg = []
ubg = []

# -- Dynamics constraints using a CasADi map --
# To vectorize the shooting constraints we “stack” the N states and controls.
# Note: Our function F expects inputs with shape (3,1) so we pack the data
# in matrices with each column corresponding to one stage.
# Prepare matrices: X0_mat will be 3 x N, U_mat will be 2 x N, and T_mat is 1 x N.
X0_mat = ca.horzcat(*X[:-1])   # each X[k] is 3x1  -> shape (3, N)
U_mat  = ca.horzcat(*U)        # shape (2, N)
T_mat  = ca.reshape( ca.vertcat(*T[:-1]), (1, N) )  # shape (1, N)

# Create a mapped version of F over N copies.
# Since our data is arranged with the mapping dimension in the second axis,
# we specify in_dims=[1, 1, 1] (i.e. mapping occurs along axis 1 in each input)
# F_map = ca.map(F, N, in_dims=[1,1,1], out_dims=[1])
F_map = F.map(N)

# Evaluate the mapped dynamics: for each stage, predict the next state.
# (Note: we divide T_mat by N since the integrator step length is dt = T[k]/N.)
X_next_pred = F_map( X0_mat, U_mat, T_mat/N )  # returns a 3 x N matrix

# The “target” states (from the decision vector) for stages 1,...,N:
X_target = ca.horzcat(*X[1:])   # shape 3 x N

# Shooting constraint: for each stage, we enforce X[k+1] - F(X[k],u[k],T[k]/N) = 0.
dyn_con = ca.reshape( X_target - X_next_pred, -1, 1 )
g.append(dyn_con)
lbg += [0]*int(dyn_con.numel())
ubg += [0]*int(dyn_con.numel())
equality += [True]*int(dyn_con.numel())

# -- Time equality constraints --
# We force all time–steps to be equal (T[k+1] - T[k] = 0 for k=0,...,N-1).
T_stack = ca.vertcat(*T)   # (N+1) x 1 vector
time_diff = T_stack[1:] - T_stack[:-1]
g.append(time_diff)
lbg += [0]*int(time_diff.numel())
ubg += [0]*int(time_diff.numel())
equality += [True]*int(time_diff.numel())

# -- Obstacle avoidance constraints --
# Here we enforce that the first two states (position) remain outside a circular obstacle.
# Obstacle center:
pos0 = ca.DM([0.2, 5])
r0 = 1
# Stack the first two entries of X[k] for k = 0,...,N-1 into a 2 x N matrix.
pos_mat = ca.horzcat(*[ X[k][0:2] for k in range(N) ])
# For each column compute the squared distance: (x - pos0_x)^2 + (y - pos0_y)^2.
# ca.sum1(·) sums along the first dimension (giving a row vector); we transpose to get a column.
obs_con = ca.transpose( ca.sum1( (pos_mat - ca.repmat(pos0, 1, N))**2 ) )
g.append(obs_con)
lbg += [r0**2]*N
ubg += [ca.inf]*N
equality += [False]*N

# -------------------------------
# 5. Define the objective and build the NLP
# -------------------------------
# We choose the final time T[N] as the cost to be minimized.
obj = T[-1]

nlp = {'x': V, 'f': obj, 'g': ca.vertcat(*g)}

# -------------------------------
# 6. Set up solver options and (optionally) code–generation options
# -------------------------------
settings = {
    'expand': True,
    'code_gen': 'None',
    'flags': [],
    'solver': 'fatrop'
}

jit_opts = {'jit': settings['code_gen'] == 'jit',
            'compiler': 'shell',
            'jit_options': {
                'flags': ['-v'] + settings['flags'],
                'compiler': 'gcc',
                'verbose': True
            }}

fatrop_opts = {
    'expand': settings['expand'],
    'fatrop': {"mu_init": 0.1},
    'structure_detection': 'auto',
    'debug': True,
    'equality': equality
}

ipopt_opts = {'expand': settings['expand']}

options = ({'ipopt': ipopt_opts, 'fatrop': fatrop_opts}[settings['solver']]) | jit_opts

opt_func = T[-1]
solver = ca.nlpsol('solver', settings['solver'], nlp, options)

# (Optionally, you could generate C code and compile a shared library as in your original code.)

nlp_time = time.time() - start_time

# -------------------------------
# 7. Solve the NLP
# -------------------------------
sol = solver(x0=x0_guess,
             lbg=lbg,
             ubg=ubg,
             lbx=lbx,
             ubx=ubx)

total_time = time.time() - start_time

width = len(str(settings))

print((width+3)*'=')
print(f'N points: {N:<{width-10}} ||')
print(f'N iters: {solver.stats()["iter_count"]:<{width-9}} ||')
print((width+3)*'=')
print(settings, '||')
print((width+3)*'=')
print(f'Time to construct NLP: {nlp_time:<{width-27}.3f} sec ||')
print(f'Solve time: {(total_time-nlp_time):<{width-16}.3f} sec ||')
print(f'Total time: {total_time:<{width-16}.3f} sec ||')
print((width+3)*'=')
