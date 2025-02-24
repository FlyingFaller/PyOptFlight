# from __future__ import annotations
from .functions import *
from .setup import *
from .boundary_objects import *
from typing import List, Callable, Dict, Union
from dataclasses import dataclass
import time

@dataclass(repr=False)
class Solution(AutoRepr):
    """Obejct to store the soltuion(s)"""
    X: List[List] ### self.config.N x 7 List of Lists of state solutions
    U: List[List] ### self.config.N x 3 List of Lists of control solutions
    stage: int ### Stage number solution belongs to
    t: List ### Times 

class Solver(AutoRepr):
    def __init__(self, 
                 body: "Body", 
                 stages: List["Stage"], 
                 config: "SolverConfig", 
                 x0: BoundaryObj, 
                 xf: BoundaryObj):
        
        self.body = body
        self.stages = stages
        self.config = config
        self.x0 = x0 ### Starting point (obj)
        self.xf = xf ### Ending point (obj)

        self.sols = []

        self.nstages = len(stages)
        self.use_atm = False if self.body.atm is None else True

        # Can pass sols to pre-initialize, but defaults to uninitialized. TODO: May remove. 
        self.initialized = False
        self.status = None
        self.success = True
        self.runtime = 0
        self.iter_count = 0
        self.nsolves = 0

        # Assign N, T_init, T_min, T_max lists using global default in config or stage value
        attributes = ['N', 'T_init', 'T_min', 'T_max']
        for attr in attributes:
            setattr(self, attr, [
                getattr(config, attr) if getattr(stage, attr) is None else getattr(stage, attr)
                for stage in stages
            ])

        self.Nu = sum(self.N) # Number of control points
        self.Nx = self.Nu + self.nstages # Number of state discretizations
        self.T = sum(self.T_init) # Total time horizon

        # Select constraints
        self.merge_constraints(force_source=config.force_constraints)

        # DEBUG
        self.extra_opts = {}
        self.nlpsolver = None
        self.nlpresult = None
        self.lam_g = None
        self.lam_x = None
        self.warm_start = False
        self.G = None
        self.V =None

    def stats(self) -> Dict:
        """Returns basic stats of overall solve"""
        return {'status': self.status,
                'success': self.success,
                'runtime': self.runtime,
                'iter_count': self.iter_count,
                'T': self.T,
                'nsolves': self.nsolves}

    def initialize_from_func(self, init_func: Callable, opts: Dict) -> None:
        """Initializes solver with sols computed by a function"""
        start_time = time.time()
        res = init_func(self, opts)
        self.runtime = time.time() - start_time

        self.sols.append(res['sols'])
        self.status = res['status']
        self.success = res['success']
        self.initialized = self.success
        self.iter_count = res['iter_count']
        self.T_init = [sol.t[-1] for sol in self.sols[-1]]
        self.T = sum(self.T_init)
        self.nsolves = len(self.sols)

    def initialize_from_sols(self, sols: List[Solution]) -> None:
        """Initializes solver with provided sols"""
        self.runtime = 0
        self.sols.append(sols)
        self.status = None
        self.success = True
        self.initialized = True
        self.iter_count = 0
        self.T_init = [sol.t[-1] for sol in self.sols[-1]]
        self.T = sum(self.T_init)
        self.nsolves = len(self.sols)

    def merge_constraints(self, force_source: Optional[str] = None) -> None:
            """
            Re-merge constraints from the global config and each stage. This method
            overwrites self.constraints with newly merged ConstraintSet objects.
            
            Parameters:
                force_source: If set to 'global', re-merge using only the global constraints.
                            If set to 'stage', re-merge using only the stage constraints.
                            If None (default), use stage constraints if defined, otherwise fallback to global.
            """
            self.constraints = [
                ConstraintSet.merge(
                    global_cs=self.config.global_constraints,
                    stage_cs=stage.constraints,
                    force_source=force_source
                )
                for stage in self.stages
            ]

    def update_constraints(self,
                           constraint_names: Union[str, List[str]],
                           new_values: Optional[Union[float, List[float]]] = None,
                           new_enables: Optional[Union[bool, List[bool]]] = None,
                           stages: Optional[List[int]] = None) -> None:
        """
        Bulk update constraints for selected stages.
        
        Parameters:
          - constraint_names: Either a single constraint name (str) or a list of names (e.g., ['q_max', 'alpha_max']).
          - new_values: A single value or a list of new values corresponding to the constraint names. 
                        If not provided, the constraint value remains unchanged.
          - new_enables: A single boolean or a list of booleans corresponding to the constraint names.
                         If not provided, the enabled flag remains unchanged.
          - stages: A list of stage indices (e.g., [0, 2, 3]) to update. If None, all stages will be updated.
        
        Raises:
          - ValueError: If list lengths of new_values or new_enables do not match the number of constraint names.
        """
        # Normalize the constraint_names to a list.
        if isinstance(constraint_names, str):
            constraint_names = [constraint_names]
        
        # Normalize new_values: if provided as a single value, expand it to the same length as constraint_names.
        if new_values is not None and not isinstance(new_values, list):
            new_values = [new_values] * len(constraint_names)
        
        # Do the same for new_enables.
        if new_enables is not None and not isinstance(new_enables, list):
            new_enables = [new_enables] * len(constraint_names)
        
        # Check that if lists were provided, they have the correct lengths.
        if new_values is not None and len(new_values) != len(constraint_names):
            raise ValueError("Length of new_values must match length of constraint_names")
        if new_enables is not None and len(new_enables) != len(constraint_names):
            raise ValueError("Length of new_enables must match length of constraint_names")
        
        # Determine which stages to update.
        stage_indices = np.array(stages)-1 if stages is not None else range(len(self.constraints))
        
        # Loop through the specified stages and update the constraints.
        for idx in stage_indices:
            cs = self.constraints[idx]
            for i, name in enumerate(constraint_names):
                # Update the constraint value if provided.
                if new_values is not None:
                    cs[name].value = new_values[i]
                # Update the enabled flag if provided.
                if new_enables is not None:
                    cs[name].enabled = new_enables[i]

    def set_all_constraints(self, enabled: bool, stage_indices: Optional[List[int]] = None) -> None:
        """
        Enable or disable all constraints.
        
        Parameters:
            enabled: The new enabled state (True to enable, False to disable).
            stage_indices: List of stage indices to update. If None, update all stages.
        """
        if stage_indices is None:
            stage_indices = range(len(self.constraints))
        else:
            stage_indices = np.array(stage_indices)-1
        for i in stage_indices:
            self.constraints[i].set_all_enabled(enabled)

    def solve(self) -> None:
        start_time = time.time()
        #######################
        ### SETUP SYMBOLICS ###
        #######################
        x = ca.SX.sym('[m, r, theta, phi, vr, omega, psi]', 7, 1)
        m, r, theta, phi, vr, omega, psi = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u = ca.SX.sym('[f, gamma, beta]', 3, 1)
        f, gamma, beta = u[0], u[1], u[2]
        T = ca.SX.sym('T')

        nx = x.size1() # Number of states (7)
        nu = u.size1() # number of control vars (3)
        # npar = self.nstages # Extra unknowns (1 for each time T)
        N_tot = sum(self.N)

        # Decision variables V has arangment [T0, x00, u00, T0, x01, u01 ... Tk, xkN, ukN,  Tk, xkN+1] ...
        V = ca.MX.sym('X', (N_tot + self.nstages)*(nx + 1) + (N_tot + self.nstages - 1) * nu) ### Stores N+1 nx, N+1 T, and N u (N is sum of all Ns)
        # Each stages X seperated out shape(nstages, (N+1), nx)

        print(f'{V.size1() = }')

        X = []
        U = []
        T_arr = []
        for k in range(0, self.nstages):
            X_stage = []
            U_stage = []
            T_stage = []
            past_N = sum(self.N[0:k])
            past_txu = (nx+nu+1)*(past_N + k) # all previous T, X, U
            for i in range(0, self.N[k] + 1):
                txu_start  = i*(nx + nu + 1)
                txu_end = (i+1)*(nx + nu + 1)
                T_stage.append(V[past_txu + txu_start])
                X_stage.append(V[past_txu + txu_start + 1 : past_txu + txu_end - nu])
                if k+1 < self.nstages or i+1 < self.N[k]+1: # If not last point of last stage
                    U_stage.append(V[past_txu + txu_start + nx + 1 : past_txu + txu_end])

            X.append(X_stage)
            U.append(U_stage)
            T_arr.append(T_stage)

        print(f'{len(T_arr) = }')
        print(f'{len(T_arr[0]) = }')
        print(f'{len(X) = }')
        print(f'{len(X[0]) = }')
        print(f'{len(U) = }')
        print(f'{len(U[0]) = }')
        # X = [[V[npar + nu*(sum(self.N[0:k])) + nx*(sum(self.N[0:k]) + k) + i*(nx+nu) : npar + nu*(sum(self.N[0:k])) + nx*(sum(self.N[0:k]) + k) + (i+1)*(nx+nu)] 
        #     for i in range(0, self.N[k])] 
        #     for k in range(0, self.nstages)]
        # All stages U lumped together: shape(nstages, N, nu)
        # U = [[V[npar + nu*sum(self.N[0:k]) + i*nu : npar + nu*sum(self.N[0:k]) + (i+1)*nu] 
        #     for i in range(0, self.N[k])]
        #     for k in range(0, self.nstages)]
        
        ##################
        ### INITIALIZE ###
        ##################

        # Initialize time states
        # x0 = self.T_init.copy()
        # lbx = self.T_min.copy()
        # ubx = self.T_max.copy()

        # Initialize empty x0, lbx, and ubx instead
        x0 = []
        lbx = []
        ubx = []

        # New equality list of fatrop constraints
        equality = []

        # Bounds from boundary points
        xb_0 = self.x0.get_xb(X[0][0], self)
        gb_0 = self.x0.get_gb(X[0][0], self)
        xb_f = self.xf.get_xb(X[-1][-1], self)
        gb_f = self.xf.get_gb(X[-1][-1], self)

        # Add constraints associated with boundary points if any, fatrop might expect gap closing first
        # Create constraint lists
        G: List = gb_0['g'] + gb_f['g']
        lbg: List = gb_0['lbg'] + gb_f['lbg']
        ubg: List = gb_0['ubg'] + gb_f['ubg']
        for (ub, lb) in zip(ubg, lbg):
            if ub == lb:
                equality.append(True)
            else:
                equality.append(False)

        print(f'{len(G) = }')
        print(f'{len(ubg) = }')
        print(f'{len(lbg) = }')


        # Free state
        # lbx_free = np.min([xb_0['lbx'], xb_f['lbx']], axis=0).tolist()
        # ubx_free = np.max([xb_0['ubx'], xb_f['ubx']], axis=0).tolist()
        # [r, theta, phi, vr, omega psi]
        lbx_free = [self.body.r_0, 0.0, -ca.pi, -ca.inf, -ca.inf, -ca.inf]
        ubx_free = [ca.inf, ca.pi, ca.pi, ca.inf, ca.inf, ca.inf]

        # Initialize control states
        # May modify to support initial/final orientation
        # Orientation is already essentially maintained through staging
        # Throttle/orientation are not provided for directly before staging whatsoever
        # for k in range(0, self.nstages):
        #     for i in range(0, self.N[k]):
        #         x0 += self.sols[-1][k].U[i]
        #         lbx += [0.0, -ca.pi, 0.0]
        #         ubx += [1.0, ca.pi,  ca.pi]

        ########################
        ### BEGIN STAGE LOOP ###
        ########################
        for k, stage in enumerate(self.stages):
            #### EOMS ###

            ### TODO: Implement Lift and Drag as polynomials ###
            ### TODO: Implement ISP and Trust variation ###
            ### TODO: Handle no atmosphere case
            ### TODO: Handle 9.81 vs 9.81e-3 in mdot and dV calcs depending on units being used
            ### also a problem in initialize.py
            v_theta = r*omega
            v_phi = r*psi*ca.sin(theta)
            rho = self.body.atm.rho_0*ca.exp(-(r - self.body.atm.rho_0)/self.body.atm.H)
            g = self.body.g_0*(self.body.r_0/r)**2
            v_phi_rel = v_phi - r*self.body.psi*ca.sin(theta)
            v_norm = ca.sqrt(vr**2 + v_theta**2 + v_phi_rel**2)
            F_drag = 0.5*rho*stage.aero.A_ref*stage.aero.C_D*v_norm/m 
            F_thrust = stage.prop.F_SL*f
            a_r = -g + F_thrust/m*ca.cos(beta) - F_drag*vr
            a_theta = -F_thrust/m*ca.sin(gamma)*ca.sin(beta) - F_drag*v_theta
            a_phi = F_thrust/m*ca.sin(beta)*ca.cos(gamma) - F_drag*v_phi_rel

            ### ODE RHS ###
            m_dot = -F_thrust/(stage.prop.Isp_SL*9.81e-3)
            r_dot = vr
            theta_dot = omega
            phi_dot = psi
            vr_dot = a_r + r*omega**2 + r*psi**2*ca.sin(theta)**2
            omega_dot = (a_theta - 2*vr*omega + r*psi**2*ca.sin(theta)*ca.cos(theta))/r
            psi_dot = (a_phi - 2*vr*psi*ca.sin(theta) - 2*r*omega*psi*ca.cos(theta))/(r*ca.sin(theta))

            ode = ca.vertcat(m_dot, r_dot, theta_dot, phi_dot, vr_dot, omega_dot, psi_dot)

            ### INTEGRATE ###
            # From 0 to N for stage k to enforce EOMs within the stage
            if self.config.integration_method in ('cvodes', 'idas', 'collocation', 'rk'):
                print(f'RUNNING {self.config.integration_method} METHOD')
                ### BUILD DAE AND INTEGRATOR ###
                dae = {'x': x, 'p': ca.vertcat(u, T), 'ode': T/self.N[k] * ode}
                dict_of_int_opts = {
                                    'cvodes': {'nonlinear_solver_iteration': 'functional', 'max_num_steps': -1},
                                    'idas': {'nonlinear_solver_iteration': 'functional', 'max_num_steps': -1},
                                    'rk': {},
                                    'collocation': {} # currently broken
                                    }
                int_opts = dict_of_int_opts[self.config.integration_method]
                I = ca.integrator('I', self.config.integration_method, dae, 0.0, 1.0, int_opts)
                ### BUILD G ###
                for i in range(0, self.N[k]): # Only grab first N u's we disregard the N+1 u
                    x_next = I(x0=X[k][i], p=ca.vertcat(U[k][i], T_arr[k][i]))
                    G.append(X[k][i+1] - x_next['xf']) # Gap closing on x
                    G.append(T_arr[k][i+1] - T_arr[k][i]) # Gap closing on T
                    lbg += (nx+1)*[0]
                    ubg += (nx+1)*[0]
                    equality += (nx+1)*[True]
                    

            elif self.config.integration_method == 'RK4':
                print(f'RUNNING {self.config.integration_method} METHOD')
                ### BUILD INTEGRATOR ###
                odef = ca.Function('f', [x, u], [ode])
                ### BUILD G ###
                for i in range(self.N[k]):
                    dt = T_arr[k][i]/self.N[k]
                    k1 = odef(X[k][i], U[k][i])
                    k2 = odef(X[k][i] + dt/2 * k1, U[k][i])
                    k3 = odef(X[k][i] + dt/2 * k2, U[k][i])
                    k4 = odef(X[k][i] + dt * k3, U[k][i])
                    x_next = X[k][i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                    G.append(X[k][i+1] - x_next) # Gap closing on x TODO:FIXME: MUST BE IN FORM X[i+1] - F(X[i]) !!!
                    G.append(T_arr[k][i+1] - T_arr[k][i]) # Gap closing on T
                    lbg += (nx+1)*[0]
                    ubg += (nx+1)*[0]
                    equality += (nx+1)*[True]
            else:
                print('Not a valid integration method!')
                return

            # Force equality between every part of state except mass during staging
            # Force equality between extra u in every stage but last one
            # That is u_N == u_N+1 (N+1 u shouldn't even exist but here we are)
            if k+1 < self.nstages: # If this is not the last stage
                G.append(X[k+1][0][1:] - X[k][-1][1:])
                lbg += (nx-1)*[0]
                ubg += (nx-1)*[0]
                equality += (nx-1)*[True]

                G.append(U[k][-1] - U[k][-2])
                lbg += nu*[0]
                ubg += nu*[0]
                equality += nu*[True]

            ### CONSTRUCT BOUNDS ON STATE AND CONSTRAINT ###
            # lbx and ubx constrained at X[0][0] and X[-1][-1] by x0 and xf get_xb
            # additional G, lbg, and ubg constrained at X[0][0] and X[-1][-1] by x0 and xf get_gb
            # inbetween x0 and xf state is constrained by x_min, x_max, u_min, u_max except mass which 
            # depends on current stage
            # mass fixed strictly at stage boundaries 
            # [m, r, theta, phi, vr, omega, psi]

            # Get m0 and mf for stage as they are useful everywhere
            m_0 = stage.m_0
            m_f = stage.m_f
            if k+1 == 1: # First stage:
                lbx_0 = [m_0, *xb_0['lbx']]
                ubx_0 = [m_0, *xb_0['ubx']]
            else: # Not first stage but beginning of new stage
                lbx_0 = [m_0, *lbx_free]
                ubx_0 = [m_0, *ubx_free]

            if k+1 == self.nstages: # Last stage:
                lbx_f = [m_f, *xb_f['lbx']] # Mass can be anywhere inbeteen m0 and mf
                ubx_f = [m_0, *xb_f['ubx']] # opt_func will ensure its closest to m0
            else: # Not last stage but end of new stage
                lbx_f = [m_f, *lbx_free]
                ubx_f = [m_f, *ubx_free]

            lbx_u = [0.0, -ca.pi, 0.0]
            ubx_u = [1.0, ca.pi,  ca.pi]

            T_min = [self.T_min[k]]
            T_max = [self.T_max[k]]
            T_init = [self.T_init[k]]

            if k+1 < self.nstages: # include extraneous u bound
                lbx += T_min + lbx_0 + lbx_u + (self.N[k]-1)*(T_min + [m_f] + lbx_free + lbx_u) + T_min + lbx_f + lbx_u
                ubx += T_max + ubx_0 + ubx_u + (self.N[k]-1)*(T_max + [m_0] + ubx_free + ubx_u) + T_max + ubx_f + ubx_u
            else: # don't include extraneous u bound
                lbx += T_min + lbx_0 + lbx_u + (self.N[k]-1)*(T_min + [m_f] + lbx_free + lbx_u) + T_min + lbx_f
                ubx += T_max + ubx_0 + ubx_u + (self.N[k]-1)*(T_max + [m_0] + ubx_free + ubx_u) + T_max + ubx_f

            for i in range(0, self.N[k]):
                x0 += T_init # Add T
                x0 += self.sols[-1][k].X[i] # Add x
                x0 += self.sols[-1][k].U[i] # Add u
            x0 += T_init # Add last T
            x0 += self.sols[-1][k].X[-1] # Add last N + 1 x
            if k+1 < self.nstages: # If not last stage 
                x0 += self.sols[-1][k].U[-1] # Duplicate last U 

        print(f'End of constructor loop time: {time.time()-start_time}')

        ### CONSTRUCT OPTIMIZATION FUNCTION ###
        opt_func = (self.stages[-1].m_0 - X[-1][-1][0])/(self.stages[-1].m_0 - self.stages[-1].m_f)

        ### SOLVE ###
        ### TODO: Proper handling of verbosity
        nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}
        # opts = {
        #     'ipopt.print_level': 5,
        #     'print_time': True,
        #     'ipopt.bound_relax_factor': self.config.bound_relax_factor,
        #     'ipopt.check_derivatives_for_naninf': 'no',
        #     'ipopt.nlp_scaling_method': self.config.nlp_scaling_method,
        #     'ipopt.tol': self.config.solver_tol,
        #     'ipopt.max_iter': self.config.max_iter,
        #     'ipopt.mumps_mem_percent': self.config.mumps_mem_percent,
        #     'expand': self.config.integration_method == 'RK4',
        # }
        # opts.update(self.extra_opts)

        # nlpsolver = ca.nlpsol(
        #     'nlpsolver', 'ipopt', nlp, opts
        # )

        fatrop_opts = {
            'expand': True,
            'fatrop': {"mu_init": 0.1},
            'ipopt.tol': 1e-4, 
            'ipopt.max_iter': 250,
            'structure_detection': 'auto',
            'debug': True,
            'equality': equality
        }
        nlpsolver = ca.nlpsol(
            'nlpsolver', 'fatrop', nlp, fatrop_opts
        )

        print(f'Construction of NLP: {time.time()-start_time}')

        ### ADD SOLUTION TO SOLUTION SET ###
        if self.warm_start:
            print('WARM STARTING')
            result = nlpsolver(x0=x0, 
                               lbx=lbx, ubx=ubx, 
                               lbg=lbg, ubg=ubg, 
                               lam_x0=self.lam_x, lam_g0=self.lam_g)
        else:
            result = nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        V_res = result['x']
        
        intersols = []
        for k in range(0, self.nstages):
            x_res = []
            u_res = []
            past_N = sum(self.N[0:k])
            past_xu = nx*(past_N + k) + nu*past_N # N+1 nx, N nu
            total_offset = self.nstages + past_xu
            for i in range(0, self.N[k]):
                xu_start  = i*(nx + nu)
                xu_end = (i+1)*(nx + nu)
                x_res.append(np.array(V_res[total_offset + xu_start : total_offset + xu_end - nu]).flatten().tolist())
                u_res.append(np.array(V_res[total_offset + xu_start + nx : total_offset + xu_end]).flatten().tolist())
                if i+1 == self.N[k]: # if last control point, add extra x so it is N+1
                    x_res.append(np.array(V_res[total_offset + xu_end: total_offset + xu_end + nx]).flatten().tolist())

            T_res = float(V_res[k])
            t_res = np.linspace(0, T_res, self.N[k] + 1).tolist()
            sol = Solution(X=x_res,
                            U=u_res,
                            stage=k+1,
                            t=t_res,
                            )
            
            intersols.append(sol)
        self.sols.append(intersols)

        ### UPDATE STATS ###
        self.status = nlpsolver.stats()['return_status']
        self.success = self.status == 'Solve_Succeeded'
        self.iter_count = nlpsolver.stats()['iter_count']
        self.T_init = [sol.t[-1] for sol in self.sols[-1]]
        self.T = sum(self.T_init)
        self.nsolves = len(self.sols)
        self.runtime = time.time() - start_time

        ### DEBUG ###
        self.nlpsolver = nlpsolver
        self.nlpresult = result
        self.lam_g = result['lam_g']
        self.lam_x = result['lam_x']
        self.G = G
        self.V = V

    def fatrop_solve(self) -> None:
        """Solver built to test fatrop solving specifically"""
        """Single stage only for now."""
        start_time = time.time()
        N = self.N[0]
        stage = self.stages[0]

        x = ca.SX.sym('[m, r, theta, phi, vr, omega, psi]', 7, 1)
        m, r, theta, phi, vr, omega, psi = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u = ca.SX.sym('[f, gamma, beta]', 3, 1)
        f, gamma, beta = u[0], u[1], u[2]

        nx = x.size1() # Number of states (7)
        nu = u.size1() # number of control vars (3)

        # Decision variables V has arangment [x, t, u, x, t, u, x, t, u, x, t...]
        V = ca.MX.sym('V', (N+1)*nx + (N+1)*1 + N*nu)
        X = [V[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)]
        U = [V[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)]
        T = [V[(nx+nu+1)*i + nx] for i in range(N+1)]

        # Bounds from boundary points
        xb_0 = self.x0.get_xb(X[0], self)
        gb_0 = self.x0.get_gb(X[0], self)
        xb_f = self.xf.get_xb(X[-1], self)
        gb_f = self.xf.get_gb(X[-1], self)

        # EOMS
        v_theta = r*omega
        v_phi = r*psi*ca.sin(theta)
        rho = self.body.atm.rho_0*ca.exp(-(r - self.body.atm.rho_0)/self.body.atm.H)
        g = self.body.g_0*(self.body.r_0/r)**2
        v_phi_rel = v_phi - r*self.body.psi*ca.sin(theta)
        v_norm = ca.sqrt(vr**2 + v_theta**2 + v_phi_rel**2)
        F_drag = 0.5*rho*stage.aero.A_ref*stage.aero.C_D*v_norm/m 
        F_thrust = stage.prop.F_SL*f
        a_r = -g + F_thrust/m*ca.cos(beta) - F_drag*vr
        a_theta = -F_thrust/m*ca.sin(gamma)*ca.sin(beta) - F_drag*v_theta
        a_phi = F_thrust/m*ca.sin(beta)*ca.cos(gamma) - F_drag*v_phi_rel

        ### ODE RHS ###
        m_dot = -F_thrust/(stage.prop.Isp_SL*9.81e-3)
        r_dot = vr
        theta_dot = omega
        phi_dot = psi
        vr_dot = a_r + r*omega**2 + r*psi**2*ca.sin(theta)**2
        omega_dot = (a_theta - 2*vr*omega + r*psi**2*ca.sin(theta)*ca.cos(theta))/r
        psi_dot = (a_phi - 2*vr*psi*ca.sin(theta) - 2*r*omega*psi*ca.cos(theta))/(r*ca.sin(theta))

        ### ODE FUNCTION AND INTEGRATOR ###
        ode = ca.vertcat(m_dot, r_dot, theta_dot, phi_dot, vr_dot, omega_dot, psi_dot)
        odef = ca.Function('f', [x, u], [ode])
        dt = ca.SX.sym("dt")
        int_dict = {'x': x, 'u': u, 'p': dt, 'ode': ode*dt}
        intg = ca.integrator('intg', 'rk',
                            int_dict, 0, 1,
                            {"simplify":True, "number_of_finite_elements": 4})

        ### GAP CLOSING ###
        G = []
        lbg = []
        ubg = []
        equality = []
        for i in range(0, N):
            dt = T[i]/N
            k1 = odef(X[i], U[i])
            k2 = odef(X[i] + dt/2 * k1, U[i])
            k3 = odef(X[i] + dt/2 * k2, U[i])
            k4 = odef(X[i] + dt * k3, U[i])
            x_next = X[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            # G.append(X[i+1] - intg(x0=X[i], u=U[i], p=T[i]/N)['xf']) # Gap closing on x TODO:FIXME: MUST BE IN FORM X[i+1] - F(X[i]) !!!
            G.append(X[i+1] - x_next)
            lbg += nx*[0]
            ubg += nx*[0]
            equality += [True]*nx

            G.append(T[i+1] - T[i]) # Gap closing on T
            lbg.append(0)
            ubg.append(0)
            equality += [True]

        # Add constraints associated with boundary points if any, fatrop might expect gap closing first
        # Create constraint lists
        G += gb_0['g'] + gb_f['g']
        lbg += gb_0['lbg'] + gb_f['lbg']
        ubg += gb_0['ubg'] + gb_f['ubg']
        for (ub, lb) in zip(gb_0['lbg'] + gb_f['lbg'], gb_0['ubg'] + gb_f['ubg']):
            if ub == lb:
                equality.append(True)
            else:
                equality.append(False)

        ### CONSTRUCT BOUNDS ON STATE ###
        x0 = []
        lbx = []
        ubx = []
        lbx_free = [self.body.r_0, 0.0, -ca.pi, -ca.inf, -ca.inf, -ca.inf]
        ubx_free = [ca.inf, ca.pi, ca.pi, ca.inf, ca.inf, ca.inf]
        m_0 = stage.m_0
        m_f = stage.m_f
        sol = self.sols[-1][0]
        T_min = self.T_min[0]
        T_max = self.T_max[0]
        T_init = self.T_init[0]
        lbx_u = [0.0, -ca.pi, 0.0]
        ubx_u = [1.0, ca.pi,  ca.pi]

        for i in range(0, N+1):
            x0 += sol.X[i]
            x0 += [T_init]
            if i == 0:
                x0 += sol.U[i]
                lbx += [m_0, *xb_0['lbx']] + [T_min] + lbx_u
                ubx += [m_0, *xb_0['ubx']] + [T_max] + ubx_u
            elif i == N:
                lbx += [m_f, *xb_f['lbx']] + [T_min]
                ubx += [m_0, *xb_f['ubx']] + [T_max]
            else:
                x0 += sol.U[i]
                lbx += [m_f, *lbx_free] + [T_min] + lbx_u
                ubx += [m_0, *ubx_free] + [T_max] + ubx_u
            
        print(f'End of constructor loop time: {time.time()-start_time}')

        ### CONSTRUCT OPTIMIZATION FUNCTION ###
        opt_func = (m_0 - X[-1][0])/(m_0 - m_f)

        ### SOLVE ###
        ### TODO: Proper handling of verbosity
        nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}

        fatrop_opts = {
            'expand': True,
            'fatrop': {"mu_init": 0.1},
            'structure_detection': 'auto',
            'debug': True,
            'equality': equality
        }
        nlpsolver = ca.nlpsol(
            'nlpsolver', 'fatrop', nlp, fatrop_opts
        )

        print(f'Construction of NLP: {time.time()-start_time}')

        ### ADD SOLUTION TO SOLUTION SET ###
        result = nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)


                





