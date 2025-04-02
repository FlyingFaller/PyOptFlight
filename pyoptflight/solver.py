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
        self.nlp_creation_time = 0
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
        # self.extra_opts = {}
        # self.nlpsolver = None
        # self.nlpresult = None
        # self.lam_g = None
        # self.lam_x = None
        # self.warm_start = False
        # self.G = None
        # self.V =None
        self.delta = 0.01 # sensitivity of f_min constraint transistion region

    def stats(self) -> Dict:
        """Returns basic stats of overall solve"""
        return {'status': self.status,
                'success': self.success,
                'runtime': self.runtime,
                'iter_count': self.iter_count,
                'T': self.T,
                'nsolves': self.nsolves,
                'nlp_creation_time': self.nlp_creation_time,
                'timestep_sizes': [self.T_init[i]/self.N[i] for i in range(self.nstages)],
                'final_mass': self.sols[-1][-1].X[-1][0]}

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

    def create_nlp(self) -> None:
        # This version will work the the structure:
        # ⬐s1⬎⬐s2⬎⬐s3⬎⬐────stage 1─────⬐⬎────stage 2────⬐⬎────stage 3─────⬎
        # [T,   T,   T,   X, U, X, U, X, U, X, U, X, U, X, U, X, U, X, U, X, U, X]
        # Written in another form:
        # ⬐s1⬎⬐s2⬎⬐s3⬎⬐───stage 1───⬐⬎───stage 2───⬐⬎───stage 3────⬎
        # [T,   T,   T,   X,   X,   X,   X,   X,    X,   X,   X,    X,   X]
        #                 ↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥↳ U ⮥
        # This means V has length sum(self.N) + 1 + self.nstages
        # Each stage has N+1 X but they share X at stage interfaces reducing problem size 
        # from sum(N+1) to sum(N)+1
        # Each stage also has a T so add nstages to total
        # This structure is not maximally sparse but it is close and it is easy to work with
        # Ideally the T's would be distributed amongst the state/input vectors as they are in 
        # FATROP

        start_time = time.time()

        x = ca.SX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u = ca.SX.sym('[f, psi, theta]', 3, 1)
        f, psi, theta = u[0], u[1], u[2]

        nx = x.size1() # Number of states (10)
        nu = u.size1() # number of control vars (3)

        V = ca.MX.sym('V', sum(self.N)*(nx + nu) + self.nstages + nx)
        T = V[0:self.nstages]
        T_sum = ca.sum1(T)
        X = []
        U = []
        G = []
        
        # all blocks start after nstages number of T's
        # each block has N nx and nu with an additional sometimes overlapping nx
        block_bounds = np.concatenate(([0], (nx+nu)*np.cumsum(self.N))) + self.nstages
        for k in range(self.nstages):
            N = self.N[k]
            Vk = V[block_bounds[k] : block_bounds[k+1] + nx]
            Uk = [Vk[(nx+nu)*i + nx : (nx+nu)*(i+1)] for i in range(N)]
            Xk = [Vk[(nx+nu)*i: (nx+nu)*(i+1) - nu] for i in range(N + 1)]
            U.append(Uk)
            X.append(Xk)

        for k, stage in enumerate(self.stages):
            ###############
            ### PHYSICS ###
            ###############    

            ### EOMS ###
            # Temp vars before more complete model comes together#
            K = 500
            C_A = -stage.aero.C_D
            C_Ny = stage.aero.C_L
            C_Nz = stage.aero.C_L

            # Constraints 
            f_min_constr = self.constraints[k].f_min
            q_constr = self.constraints[k].max_q
            alpha_constr = self.constraints[k].max_alpha
            body_rate_y_constr = self.constraints[k].max_body_rate_y
            body_rate_z_constr = self.constraints[k].max_body_rate_z
            tau_constr = self.constraints[k].max_tau

            # Apply f_min constraint
            if f_min_constr.enabled and f_min_constr.value is not None:
                f_min = f_min_constr.value
            else:
                f_min = 0

            # Supporting Definitions #
            h = ca.sqrt(px**2 + py**2 + pz**2) - self.body.r_0 # Altitude
            F_max = stage.prop.F_vac + (stage.prop.F_SL - stage.prop.F_vac)*ca.exp(-h/self.body.atm.H) # Max thrust
            # F_eff = F_max*f/(1 + ca.exp(-K*(f - f_min))) # Effective thrust
            f_eff = f - f*ca.fmax(0, ca.fmin(1, (f_min - f)/self.delta))
            F_eff = f_eff*F_max
            Isp = stage.prop.Isp_vac + (stage.prop.Isp_SL - stage.prop.Isp_vac)*ca.exp(-h/self.body.atm.H) # Isp
            g = -self.body.g_0*self.body.r_0**2*(px**2 + py**2 + pz**2)**(-3/2)*ca.vertcat(px, py, pz) # gravity vector
            rho = self.body.atm.rho_0*ca.exp(-h/self.body.atm.H) # denisty
            v_rel = ca.vertcat(vx + self.body.omega_0*py, vy - self.body.omega_0*px, vz) # atmosphere relative velocity

            # body fram basis vectors
            ebx = ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
            # eby = ca.vertcat(-ca.sin(psi), ca.cos(psi), 0)
            # ebz = ca.vertcat(ca.cos(psi)*ca.sin(theta), ca.sin(psi)*ca.sin(theta), ca.cos(theta))
            
            m_dot = -F_eff/(Isp*9.81e-3)
            px_dot = vx
            py_dot = vy
            pz_dot = vz
            # vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[0] + C_Ny*eby[0] + C_Nz*ebz[0])
            # vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[1] + C_Ny*eby[1] + C_Nz*ebz[1])
            # vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[2] + C_Ny*eby[2] + C_Nz*ebz[2])
            # Drag only version if needed in testing
            vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*stage.aero.A_ref*ca.norm_2(v_rel)*C_A*v_rel[0]
            vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*stage.aero.A_ref*ca.norm_2(v_rel)*C_A*v_rel[1]
            vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*stage.aero.A_ref*ca.norm_2(v_rel)*C_A*v_rel[2]

            ###############################
            ### ODE FUNC AND INTEGRATOR ###
            ###############################

            ode = ca.vertcat(m_dot, px_dot, py_dot, pz_dot, vx_dot, vy_dot, vz_dot)
            F_ode = ca.Function('F_ode', [x, u], [ode])
            # All integrators need x, u, dt (symbolics) and should return x_next
            dt = ca.SX.sym("dt")
            if self.config.integration_method == 'RK4': # Implement more int methods later RK4
                k1 = F_ode(x, u)
                k2 = F_ode(x + dt/2 * k1, u)
                k3 = F_ode(x + dt/2 * k2, u)
                k4 = F_ode(x + dt * k3, u)
                x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                F_int = ca.Function('F_int', [x, u, dt], [x_next])
            elif self.config.integration_method == 'cvodes':
                dae = {'x': x, 'u':u, 'p': dt, 'ode': dt*F_ode(x, u)}
                int_opts = {'nonlinear_solver_iteration': 'functional'}
                I = ca.integrator('I', 'cvodes', dae, 0.0, 1.0, int_opts)
                x_mx = ca.MX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
                u_mx = ca.MX.sym('[f, psi, theta]', 3, 1)
                dt_mx = ca.MX.sym('dt_mx')
                F_int = ca.Function('F_int', [x_mx, u_mx, dt_mx], [I(x0=x_mx, u=u_mx, p=dt_mx)['xf']])
            else:
                raise NotImplementedError(f'{self.config.integration_method} is not an implmented integrator.')
            
            ##############################
            ### CONSTRAINT APPLICATION ###
            ##############################

            N = self.N[k]
            for i in range(N): # iterate through N stage nodes
                # gap closing
                if i+1 == N and k+1 < self.nstages: # if last node of a booster stage
                    # stage i+1 mass plus stage i empty mass must equal integration of stage i mass flow
                    m_e = self.stages[k].m_f - self.stages[k+1].m_0 # stage empty mass
                    G.append(X[k][i+1][0] + m_e - F_int(X[k][i], U[k][i], T[k]/N)[0])
                    G.append(X[k][i+1][1:] - F_int(X[k][i], U[k][i], T[k]/N)[1:])

                else: # all other nodes
                    G.append(X[k][i+1] - F_int(X[k][i], U[k][i], T[k]/N))

                if i == 0 and k == 0: # if first node and first stage
                    # initial constraint placed here for sparcity
                    ge_0 = self.x0.get_ge(X[0][0], U[0][0], T_sum, self)
                    G += ge_0['g']
                else: # all other nodes
                    # add path constraints
                    # do not let path dip below planet radius
                    G.append(ca.sumsqr(ca.vertcat(X[k][i][1:4])) - self.body.r_0**2)

                    if f_min_constr.enabled and f_min_constr.value is not None:
                        G.append((U[k][i][0] - f_min_constr.value + self.delta)*(U[k][i][0] - f_min_constr.value))

                    # constraints currently do not consider stage interface!
                    xi = X[k][i]
                    px, py, pz = xi[1], xi[2], xi[3]
                    vx, vy, vz = xi[4], xi[5], xi[6]
                    if q_constr.enabled and q_constr.value is not None: # max q
                        # Make h, rho, v_rel casadi functions?
                        h = ca.sqrt(px**2 + py**2 + pz**2) - self.body.r_0
                        rho = self.body.atm.rho_0*ca.exp(-h/self.body.atm.H)
                        v_rel_2 = (vx + self.body.omega_0*py)**2 + (vy - self.body.omega_0*px)**2 + vz**2
                        q = 0.5*rho*v_rel_2
                        G.append(q_constr.value - q) # Must be >= 0

                    if alpha_constr.enabled and alpha_constr.value is not None: # max AoA
                        psi, theta = U[k][i][1], U[k][i][2]
                        ebx = ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
                        v_rel = ca.vertcat(vx + self.body.omega_0*py, vy - self.body.omega_0*px, vz)
                        v_rel = -v_rel if self.config.landing else v_rel
                        cos_alpha = ca.dot(v_rel, ebx)/ca.norm_2(v_rel)
                        G.append(cos_alpha - ca.cos(alpha_constr.value)) # Must be >= 0
                
                # rate constraints (temporary)
                if i+1 < N: # if not last node
                    dt = T[k]/N
                    if tau_constr.enabled and tau_constr.value is not None:
                        # G.append(tau_constr.value - ca.fabs(U[k][i+1][0]-U[k][i][0])/dt) # Must be >= 0
                        G.append((U[k][i+1][0]-U[k][i][0])/dt)
                    if body_rate_y_constr.enabled and body_rate_y_constr.value is not None:
                        # G.append(body_rate_y_constr.value*ca.cos(U[k][i][2]) - ca.fabs(U[k][i+1][1]-U[k][i][1])/dt) # Must be >= 0
                        G.append((U[k][i+1][1]-U[k][i][1])/dt*ca.cos(U[k][i][2]))
                    if body_rate_z_constr.enabled and body_rate_z_constr.value is not None:
                        # G.append(body_rate_z_constr.value - ca.fabs(U[k][i+1][2]-U[k][i][2])/dt) # Must be >= 0
                        G.append((U[k][i+1][2]-U[k][i][2])/dt)

        # final constraint placed here for sparcity
        ge_f = self.xf.get_ge(X[-1][-1], U[-1][-1], T_sum, self)
        G += ge_f['g']

        # Optimization function
        opt_func = (self.stages[-1].m_0 - X[-1][-1][0])/(self.stages[-1].m_0 - self.stages[-1].m_f)

        # Create solver
        nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}
        ipopt_opts = {
            'expand': self.config.integration_method == 'RK4',
            'ipopt.nlp_scaling_method': 'none',
            'ipopt.tol': self.config.solver_tol,
            'ipopt.max_iter': self.config.max_iter
            }
        nlpsolver = ca.nlpsol('nlpsolver', 'ipopt', nlp, ipopt_opts)
        self.nlpsolver = nlpsolver
        self.nlp_creation_time = time.time() - start_time        

    def solve_nlp(self) -> None:
        start_time = time.time()
        if self.nlpsolver is None:
            raise Exception('NLP Solver must be created with create_nlp()')
        elif not self.initialized:
            raise Exception('Solver must be initialized with a guess solution.')
        
        nx = 7
        nu = 3
        x0, lbx, ubx, lbg, ubg = [], [], [], [], []

        # create x0
        x0 += self.T_init
        for k in range(self.nstages):
            for i in range(self.N[k]):
                x0 += self.sols[-1][k].X[i] + self.sols[-1][k].U[i]
        x0 += self.sols[-1][-1].X[-1]

        # create lbg ubg
        for k in range(self.nstages):
            f_min_constr = self.constraints[k].f_min
            q_constr = self.constraints[k].max_q
            alpha_constr = self.constraints[k].max_alpha
            body_rate_y_constr = self.constraints[k].max_body_rate_y
            body_rate_z_constr = self.constraints[k].max_body_rate_z
            tau_constr = self.constraints[k].max_tau
            for i in range(self.N[k]):
                # gap closing:
                lbg += nx*[0]
                ubg += nx*[0]

                if i == 0 and k == 0:
                    # initial constraints
                    gb_0 = self.x0.get_gb(self)
                    lbg += gb_0['lbg']
                    ubg += gb_0['ubg']
                else:
                    # radius constraint
                    lbg.append(0)
                    ubg.append(ca.inf)
                    if f_min_constr.enabled and f_min_constr.value is not None:
                        lbg.append(0)
                        ubg.append(1)
                    if q_constr.enabled and q_constr.value is not None: # max q
                        lbg.append(0)
                        ubg.append(q_constr.value)
                    if alpha_constr.enabled and alpha_constr.value is not None: # max AoA
                        lbg.append(0)
                        ubg.append(2)
                if i+1 < self.N[k]: # if not last node
                    if tau_constr.enabled and tau_constr.value is not None:
                        lbg.append(-tau_constr.value)
                        ubg.append(tau_constr.value)
                    if body_rate_y_constr.enabled and body_rate_y_constr.value is not None:
                        lbg.append(-body_rate_y_constr.value)
                        ubg.append(body_rate_y_constr.value)
                    if body_rate_z_constr.enabled and body_rate_z_constr.value is not None:
                        lbg.append(-body_rate_z_constr.value)
                        ubg.append(body_rate_z_constr.value)

        # final constraint
        gb_f = self.xf.get_gb(self)
        lbg += gb_f['lbg']
        ubg += gb_f['ubg']

        # create lbx ubx
        free_bound = FreeBound()
        xb_free = free_bound.get_xb(self)
        ub_free = free_bound.get_ub(self)
        lbx += self.T_min
        ubx += self.T_max
        for k, stage in enumerate(self.stages):
            N = self.N[k]
            m_0 = stage.m_0
            m_f = stage.m_f
            f_min_constr = self.constraints[k].f_min
            if f_min_constr.enabled and f_min_constr.value is not None:
                lbf = max(f_min_constr.value - self.delta, 0)
            else:
                lbf = 0
            
            # bounds on first node
            if k == 0:
                xb_0 = self.x0.get_xb(self)
                ub_0 = self.x0.get_ub(self)
                lbx += [m_0] + xb_0['lbx'] + [lbf] + ub_0['lbu']
                ubx += [m_0] + xb_0['ubx'] + [1]   + ub_0['ubu']
            else: # stage interface nodes are here vvv
                lbx += [m_0] + xb_free['lbx'] + [lbf] + ub_free['lbu']
                ubx += [m_0] + xb_free['ubx'] + [1]   + ub_free['ubu']

            # bounds on next N-1 nodes (N total at this point)
            if k + 1 < self.nstages: # if not last stage
                lbx += (N-1)*([m_f] + xb_free['lbx'] + [lbf] + ub_free['lbu'])
                ubx += (N-1)*([m_0] + xb_free['ubx'] + [1]   + ub_free['ubu'])
            else: # Last stage gets extra xb and special ub 
                lbx += (N-2)*([m_f] + xb_free['lbx'] + [lbf] + ub_free['lbu'])
                ubx += (N-2)*([m_0] + xb_free['ubx'] + [1]   + ub_free['ubu'])

                # final constraint
                xb_f = self.xf.get_xb(self)
                ub_f = self.xf.get_ub(self)
                lbx += [m_f] + xb_free['lbx'] + [lbf] + ub_f['lbu'] + [m_f] + xb_f['lbx']
                ubx += [m_0] + xb_free['ubx'] + [1]   + ub_f['ubu'] + [m_0] + xb_f['ubx']

        # SOLVE #
        result = self.nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # # PARSE RESULTS #
        intersols = []
        V_res = np.array(result['x']).flatten()
        block_bounds = np.concatenate(([0], (nx+nu)*np.cumsum(self.N))) + self.nstages
        for k in range(self.nstages):
            N = self.N[k]
            Vk = V_res[block_bounds[k] : block_bounds[k+1] + nx]
            t_res = np.linspace(0, V_res[k], N + 1)
            Uk = np.array([Vk[(nx+nu)*i + nx : (nx+nu)*(i+1)] for i in range(N)])
            Xk = np.array([Vk[(nx+nu)*i: (nx+nu)*(i+1) - nu] for i in range(N + 1)])

            # add back in structural mass at end of stage
            if k + 1 < self.nstages:
                m_e = self.stages[k].m_f - self.stages[k+1].m_0 # stage empty mass
                Xk[-1][0] += m_e

            sol = Solution(
                X=Xk.tolist(),
                U=Uk.tolist(),
                stage=k+1,
                t=t_res.tolist(),
                )
            intersols.append(sol)
        self.sols.append(intersols)

        ### UPDATE STATS ###
        self.status = self.nlpsolver.stats()['return_status']
        self.success = self.status == 'Solve_Succeeded'
        self.iter_count = self.nlpsolver.stats()['iter_count']
        self.T_init = [sol.t[-1] for sol in self.sols[-1]]
        self.T = sum(self.T_init)
        self.nsolves = len(self.sols)
        self.runtime = time.time() - start_time