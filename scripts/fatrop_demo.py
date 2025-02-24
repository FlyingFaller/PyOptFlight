print('\nRunning demo!\n')

import time as time
import casadi as ca

start_time = time.time()

pos = ca.MX.sym('pos',2) # position
theta = ca.MX.sym('theta') # angle

delta = ca.MX.sym('delta') # rate of change of angle or equivalent
V     = ca.MX.sym('V') # velocity

# States
x = ca.vertcat(pos,theta) # x = symbolic states

# Controls
u = ca.vertcat(delta,V) # u = symbolic controls

L = 1 # L? Length of bike

# ODE rhs
# Bicycle model
# (S. LaValle. Planning Algorithms. Cambridge University Press, 2006, pp. 724–725.)

ode = ca.vertcat(V*ca.vertcat(ca.cos(theta), ca.sin(theta)), V/L*ca.tan(delta)) # ODE

# Discretize system
dt = ca.MX.sym("dt")
sys = {}
sys["x"] = x
sys["u"] = u
sys["p"] = dt
sys["ode"] = ode*dt # Time scaling

# integrator from 0 to 1 scaled by dt
intg = ca.integrator('intg','rk',sys,0,1,{"simplify":True, "number_of_finite_elements": 4})

# create a function that runs the integrator when x, u, dt are supplied returning xnext
F = ca.Function('F',[x,u,dt],[intg(x0=x,u=u,p=dt)["xf"]],["x","u","dt"],["xnext"])

# Number of states and controls (3, 2)
nx = x.numel()
nu = u.numel()

f = 0 # Objective
x = [] # List of decision variable symbols !!!x is no longer the symbolic states!!!
lbx = []; ubx = [] # Simple bounds
x0 = [] # Initial value
g = [] # Constraints list
lbg = []; ubg = [] # Constraint bounds
equality = [] # Boolean indicator helping structure detection
p = [] # Parameters
p_val = [] # Parameter values

N = 50
T0 = 10

X = [] # all states (symbolic)
U = [] # all controls (symbolic)
T = [] # all times (symbolic)
for k in range(N+1):
    sym = ca.MX.sym("X",nx) # sym is now essentially what old x was
    x.append(sym) # append nx symbolic states to x
    X.append(sym) # append nx symbolic states to X what is really the difference between x and X anymore??
    x0.append(ca.vertcat(0,k*T0/N,ca.pi/2)) # some sort of init guess for x0 which is only states for now 
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
    sym = ca.MX.sym("T") # god help us he's reusing sym as a variable name
    x.append(sym)
    T.append(sym)
    x0.append(T0)
    lbx.append(0);ubx.append(ca.inf)
    
    # add in control initial and limits N times so structure looks like [x, T, u], [x, T, u], [x, T, u], ... [x, T]
    if k<N:
        sym = ca.MX.sym("U",nu)
        x.append(sym)
        U.append(sym)
        x0.append(ca.vertcat(0,1))
        lbx.append(-ca.pi/6);ubx.append(ca.pi/6) # -pi/6 <= delta<= pi/6
        lbx.append(0);ubx.append(1) # 0 <= V<=1

# could everything above be done is like 6 lines and be more readable? yes. Is this approach better? no....wait a minute.

# Round obstacle
pos0 = ca.vertcat(0.2,5)
r0 = 1

# setting the initial position looks like
# X0 = ca.MX.sym("X0",nx)
# p.append(X0)
# p_val.append(ca.vertcat(0,0,pi/2))

f = sum(T) # Time Optimal objective
for k in range(N):
    # Multiple shooting gap-closing constraint
    g.append(X[k+1]-F(X[k],U[k],T[k]/N))
    lbg.append(ca.DM.zeros(nx,1))
    ubg.append(ca.DM.zeros(nx,1))
    equality += [True]*nx # added equality constraint for gap closing 
    
    # each time is force equal to eachother by similar gap closing means 
    g.append(T[k+1]-T[k])
    lbg.append(0);ubg.append(0)
    equality += [True]
    
    # weird additional constraint on initial pos. Why not just put this in lbx/ubx?
    # if k==0:
    #     # Initial constraints
    #     g.append(X[0]-X0)
    #     lbg.append(ca.DM.zeros(nx,1))
    #     ubg.append(ca.DM.zeros(nx,1))
    #     equality += [True]*nx
        
    # Obstacle avoidance
    pos = X[k][:2]
    g.append(ca.sumsqr(pos-pos0))
    lbg.append(r0**2);ubg.append(ca.inf)
    equality += [False]
    
    # same as initial pos constraint, does the order matter here that means they must be in the loop? experiment with this
    # if k==N-1:
    #     # Final constraints
    #     g.append(X[-1][:2])
    #     lbg.append(ca.vertcat(0,10));ubg.append(ca.vertcat(0,10))
    #     equality += [True,True]

print(X[0][0])

# Add some regularization
# for k in range(N+1):
    # f += X[k][0]**2

# Solve the problem

nlp = {}
nlp["f"] = T[-1]
nlp["g"] = ca.vcat(g)
nlp["x"] = ca.vcat(x)
# nlp["p"] = ca.vcat(p)

options = {}
options["expand"] = True
options["fatrop"] = {"mu_init": 0.1}
options["structure_detection"] = "auto"
options["debug"] = False
options["equality"] = equality

# (codegen of helper functions)
options["jit"] = True
# options["jit_temp_suffix"] = False
options['compiler'] = 'shell'
options["jit_options"] = {"flags": ["-O3", "-Ofast", "-march=native"],
                        #   "compiler": "wsl gcc",
                          "verbose": True}

solver = ca.nlpsol('solver', "fatrop", nlp, options)
# solver = ca.nlpsol('solver', 'ipopt', nlp, {'expand': True})

nlp_time = time.time() - start_time

res = solver(x0 = ca.vcat(x0),
    lbg = ca.vcat(lbg),
    ubg = ca.vcat(ubg),
    lbx = ca.vcat(lbx),
    ubx = ca.vcat(ubx),
    # p = ca.vcat(p_val)
)

print(f'Time to construct NLP: {nlp_time:.3f}')
print(f'Total time: {(time.time() - start_time):.3f}')
