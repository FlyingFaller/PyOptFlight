print('\nRunning demo!\n')

import casadi as ca

x = ca.SX.sym('[px, py, theta]', 3, 1)
u = ca.SX.sym('[delta, vel]', 2, 1)

px, py, theta = x[0], x[1], x[2]

delta, vel = u[0], u[1]

L = 1 # L? Length of bike

# ODE rhs
# Bicycle model
# (S. LaValle. Planning Algorithms. Cambridge University Press, 2006, pp. 724–725.)

px_dot = vel*ca.cos(theta)
py_dot = vel*ca.sin(theta)
theta_dot = vel/L*ca.tan(delta)

ode = ca.vertcat(px_dot, py_dot, theta_dot)

# Discretize system
dt = ca.SX.sym("dt")

int_dict = {'x': x, 'u': u, 'p': dt, 'ode': ode*dt}

# integrator from 0 to 1 scaled by dt
intg = ca.integrator('intg', 'rk',
                     int_dict, 0, 1,
                     {"simplify":True, "number_of_finite_elements": 4})

# Number of states and controls (3, 2)
nx = x.numel()
nu = u.numel()

#######################################################
### EVERYTHING BELOW THIS POINT IS *NOW LESS FUCKED ###
#######################################################

x0 = [] # Initial value
lbx = []
ubx = [] # Simple bounds

G = [] # Constraints list
lbg = []
ubg = [] # Constraint bounds
equality = [] # Boolean indicator helping structure detection

N = 20
T0 = 10

V = ca.MX.sym('V', (N+1)*nx + (N+1)*1 + N*nu)
X = [V[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)]
U = [V[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)]
T = [V[(nx+nu+1)*i + nx] for i in range(N+1)]
# nx = 3
# nu = 2
#  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15
# [x1, x2, x3, t1, u1, u2, x1, x2, x3, t1, u1, u2, x1, x2, x3, t1]

for k in range(N+1):
    x0 += [0, k*T0/N, ca.pi/2]
    if k == 0:
        lbx += [0, 0, 0]
        ubx += [0, 0, 0]
    elif k+1 == N+1:
        lbx += [0, 10, -ca.inf]
        ubx += [0, 10, ca.inf]
    else:
        lbx += 3*[-ca.inf]
        ubx += 3*[ca.inf] # Looks like the states are unbounded essentially
    
    # this whole bit is about adding a T for every state
    x0.append(T0)
    lbx.append(0)
    ubx.append(ca.inf)
    
    # add in control initial and limits N times so structure looks like [x, T, u], [x, T, u], [x, T, u], ... [x, T]
    if k<N:
        x0 += [0, 1]
        lbx += [-ca.pi/6, 0]
        ubx += [ca.pi/6, 1]

# could everything above be done is like 6 lines and be more readable? yes. Is this approach better? no....wait a minute.

# Round obstacle
pos0 = ca.vertcat(0.2, 5)
r0 = 1

f = sum(T) # Time Optimal objective
for k in range(N):
    # Multiple shooting gap-closing constraint
    G.append(X[k+1] - intg(x0=X[k], u=U[k], p=T[k]/N)['xf']) # MUST BE IN FORM X[i+1] = F(X[i]) !!!
    lbg += nx*[0]
    ubg += nx*[0]

    equality += [True]*nx # added equality constraint for gap closing 
    
    # each time is force equal to eachother by similar gap closing means 
    G.append(T[k+1]-T[k]) # MUST BE IN FORM X[i+1] - F(X[i]) !!!
    lbg.append(0)
    ubg.append(0)
    equality += [True]

    # Obstacle avoidance
    pos = X[k][:2]
    G.append(ca.sumsqr(pos-pos0))
    lbg.append(r0**2)
    ubg.append(ca.inf)
    equality += [False]

# Solve the problem

fatrop_opts = {
    'expand': True,
    'fatrop': {"mu_init": 0.1},
    'structure_detection': 'auto',
    'debug': True,
    'equality': equality
}

opt_func = T[-1]
nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}
solver = ca.nlpsol('solver',"fatrop", nlp, fatrop_opts)

res = solver(x0 = x0,
    lbg = lbg,
    ubg = ubg,
    lbx = lbx,
    ubx = ubx,
)
