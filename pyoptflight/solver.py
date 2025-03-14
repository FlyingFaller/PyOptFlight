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
                'nsolves': self.nsolves,
                'nlp_creation_time': self.nlp_creation_time}

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
        # NLP requires V, opt_func, G, equality
        start_time = time.time()
        x = ca.SX.sym('[m, px, py, pz, vx, vy, vz, f, psi, theta]', 10, 1)
        m, px, py, pz, vx, vy, vz, f, psi, theta = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
        u = ca.SX.sym('[tau, r, q]', 3, 1)
        tau, r, q = u[0], u[1], u[2]

        nx = x.size1() # Number of states (10)
        nu = u.size1() # number of control vars (3)

        V = []
        X = []
        U = []
        T = []
        G = []
        E = []
        for k in range(self.nstages):
            N = self.N[k]
             # Decision vars for this stage
            Vk = ca.MX.sym('V', (N+1)*nx + (N+1)*1 + N*nu)
            Xk = [Vk[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)]
            Tk = [Vk[(nx+nu+1)*i + nx] for i in range(N+1)]
            Uk = [Vk[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)]
            V.append(Vk)
            X.append(Xk)
            T.append(Tk)
            U.append(Uk)
            
        for k, stage in enumerate(self.stages):
            ### EOMS ###
            # Temp vars before more complete model comes together#
            f_min = 0
            K = 100
            C_A = -stage.aero.C_D
            C_Ny = stage.aero.C_L
            C_Nz = stage.aero.C_L

            # Supporting Definitions #
            h = ca.sqrt(px**2 + py**2 + pz**2) - self.body.r_0 # Altitude
            # F_max = stage.prop.F_vac + (stage.prop.F_SL - stage.prop.F_vac)*ca.exp(-h/self.body.atm.H) # Max thrust
            # F_eff = F_max*f/(1 + ca.exp(-K*(f - f_min))) # Effective thrust
            F_eff = stage.prop.F_SL*f
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
            f_dot = tau
            psi_dot = r/(ca.fabs(ca.cos(theta)) + 1e-6)
            theta_dot = q

            ### ODE FUNC AND INTEGRATOR ###
            ode = ca.vertcat(m_dot, px_dot, py_dot, pz_dot, vx_dot, vy_dot, vz_dot, f_dot, psi_dot, theta_dot)
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
                x_mx = ca.MX.sym('[m, px, py, pz, vx, vy, vz, f, psi, theta]', 10, 1)
                u_mx = ca.MX.sym('[tau, r, q]', 3, 1)
                dt_mx = ca.MX.sym('dt_mx')
                F_int = ca.Function('F_int', [x_mx, u_mx, dt_mx], [I(x0=x_mx, u=u_mx, p=dt_mx)['xf']])
            else:
                raise NotImplementedError(f'{self.config.integration_method} is not an implmented integrator.')

            N = self.N[k]
            for i in range(N):
                # Do gap closing ig?
                G.append(X[k][i+1] - F_int(X[k][i], U[k][i], T[k][i]/N))
                G.append(T[k][i+1] - T[k][i])
                E += (nx + 1)*[True]

                if i == 0 and k == 0: # Has to be here to give fatrop correct C, D matrix structure
                    # Add in initial constraints
                    gb_0 = self.x0.get_ge(X[0][0], T, self)
                    G += gb_0['g']
                    E += gb_0['e']
                else:
                    # path constraints, make sure vehicle does not go below planet ig
                    G.append(ca.sumsqr(ca.vertcat(X[k][i][1:4])) - self.body.r_0**2)
                    E.append(False)

            if k+1 < self.nstages:
                # Continuity between stages
                G.append(X[k+1][0][0] - X[k][-1][0] - (self.stages[k+1].m_0 - self.stages[k].m_f)) # mass
                G.append(X[k+1][0][1:10] - X[k][-1][1:10]) # pos/vel/ctrl
                E += (nx)*[True]

                # path constraints, make sure vehicle does not go below planet ig
                G.append(ca.sumsqr(ca.vertcat(X[k][-1][1:4])) - self.body.r_0**2)
                E.append(False)

        # Add in final constraints
        gb_f = self.xf.get_ge(X[-1][-1], T, self)
        G += gb_f['g']
        E += gb_f['e']

        # Optimization function
        opt_func = (self.stages[-1].m_0 - X[-1][-1][0])/(self.stages[-1].m_0 - self.stages[-1].m_f)

        # Create solver
        nlp = {'x': ca.vertcat(*V), 'f': opt_func, 'g': ca.vertcat(*G)}
        # nlp['scaling'] = {'x': x_scale, 'g': g_scale}

        if self.extra_opts.get('solver') == 'ipopt':
            ipopt_opts = {
                'expand': self.config.integration_method == 'RK4',
                'ipopt.nlp_scaling_method': 'gradient-based',
                'ipopt.tol': self.config.solver_tol,
                'ipopt.max_iter': 5000
            }
            nlpsolver = ca.nlpsol(
                'nlpsolver', 'ipopt', nlp, ipopt_opts
                )
        else:
            fatrop_opts = {
                'expand': self.config.integration_method == 'RK4',
                'fatrop': {"mu_init": 0.1},
                'structure_detection': 'auto',
                'debug': True,
                'equality': E,
            }
            nlpsolver = ca.nlpsol(
                'nlpsolver', 'fatrop', nlp, fatrop_opts
            )
        self.nlpsolver = nlpsolver
        self.nlp_creation_time = time.time() - start_time

    def solve_nlp(self) -> None:
        start_time = time.time()
        if self.nlpsolver is None:
            raise Exception('NLP Solver must be created with create_nlp()')
        elif not self.initialized:
            raise Exception('Solver must be initialized with a guess solution.')
        
        # maybe move to solver properties later?
        nx = 10 
        nu = 3
        x0, lbx, ubx, lbg, ubg = [], [], [], [], []
        # create x0
        for k in range(self.nstages):
            for i in range(self.N[k]):
                x0 += self.sols[-1][k].X[i] + [self.sols[-1][k].t[-1]] + self.sols[-1][k].U[i]
            x0 += self.sols[-1][k].X[-1] + [self.sols[-1][k].t[-1]]
                
        # create lbg ubg
        for k in range(self.nstages):
            for i in range(self.N[k]):
                # gap closing, physics and time have to match
                lbg += (nx+1)*[0]
                ubg += (nx+1)*[0]

                if i == 0 and k == 0:
                    # initial constraints lbg ubg
                    gb_0 = self.x0.get_gb(self)
                    lbg += gb_0['lbg']
                    ubg += gb_0['ubg']
                else:
                    # path constraints
                    lbg.append(0)
                    ubg.append(109225)

            if k+1 < self.nstages:
                # stage continuity
                lbg += nx*[0]
                ubg += nx*[0]

                # path constraints
                lbg.append(0)
                ubg.append(109225)


        # final constraint lbg ubg
        gb_f = self.xf.get_gb(self)
        lbg += gb_f['lbg']
        ubg += gb_f['ubg']

        # create lbx ubx
        # free states, may adjust later
        ubx_free = 6*[ca.inf] + [1, ca.pi, ca.pi/2]
        lbx_free = 6*[-ca.inf] + [0, -ca.pi, -ca.pi/2]
        ubu_free = [0.1, 0.05, 0.05]
        lbu_free = [-0.1, -0.05, -0.05]
        for k, stage in enumerate(self.stages):
            m_0 = stage.m_0
            m_f = stage.m_f
            T_min = self.T_min[k]
            T_max = self.T_max[k]

            if k+1 == 1:
                # first node first stage
                xb_0 = self.x0.get_xb(self)
                lbx += [m_0] + xb_0['lbx'] + [T_min] + lbu_free
                ubx += [m_0] + xb_0['ubx'] + [T_max] + ubu_free
            else:
                # first node other stages
                lbx += [m_0] + lbx_free + [T_min] + lbu_free
                ubx += [m_0] + ubx_free + [T_max] + ubu_free

            for i in range(self.N[k]-1):
                scaling_factor = 100 - 99*i/(self.N[k]-2)
                lbx += [m_f] + lbx_free + [T_min] + (scaling_factor*np.array(lbu_free)).tolist()
                ubx += [m_0] + ubx_free + [T_max] + (scaling_factor*np.array(ubu_free)).tolist()

            if k+1 == self.nstages:
                # last node last stage
                xb_f = self.xf.get_xb(self)
                lbx += [m_f] + xb_f['lbx'] + [T_min]
                ubx += [m_0] + xb_f['ubx'] + [T_max]
            else:
                # last node other stages
                lbx += [m_f] + lbx_free + [T_min]
                ubx += [m_f] + ubx_free + [T_max]

        # SOLVE #
        result = self.nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # PARSE RESULTS #
        V_res = np.array(result['x']).flatten()
        N_arr = np.array(self.N)
        blocks = np.concatenate(([0], np.cumsum((N_arr+1)*(nx+1) + N_arr*nu)))
        intersols = []
        for k in range(self.nstages):
            N = self.N[k]
            Vk = V_res[blocks[k] : blocks[k+1]]
            Xk = np.array([Vk[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)])
            Tk = np.array([Vk[(nx+nu+1)*i + nx] for i in range(N+1)])
            Uk = np.array([Vk[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)])
            t_res = np.linspace(0, Tk[-1], N + 1)
            sol = Solution(X=Xk.tolist(),
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


    def create_nlp_2(self) -> None:
        # NLP requires V, opt_func, G, equality
        start_time = time.time()
        x = ca.SX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        u = ca.SX.sym('[f, psi, theta]', 3, 1)
        f, psi, theta = u[0], u[1], u[2]

        nx = x.size1() # Number of states (10)
        nu = u.size1() # number of control vars (3)

        V = []
        X = []
        U = []
        T = []
        G = []
        E = []
        for k in range(self.nstages):
            N = self.N[k]
             # Decision vars for this stage
            Vk = ca.MX.sym('V', (N+1)*nx + (N+1)*1 + N*nu)
            Xk = [Vk[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)]
            Tk = [Vk[(nx+nu+1)*i + nx] for i in range(N+1)]
            Uk = [Vk[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)]
            V.append(Vk)
            X.append(Xk)
            T.append(Tk)
            U.append(Uk)
            
        for k, stage in enumerate(self.stages):
            ### EOMS ###
            # Temp vars before more complete model comes together#
            f_min = 0
            K = 100
            C_A = -stage.aero.C_D
            C_Ny = stage.aero.C_L
            C_Nz = stage.aero.C_L

            # Supporting Definitions #
            h = ca.sqrt(px**2 + py**2 + pz**2) - self.body.r_0 # Altitude
            # F_max = stage.prop.F_vac + (stage.prop.F_SL - stage.prop.F_vac)*ca.exp(-h/self.body.atm.H) # Max thrust
            # F_eff = F_max*f/(1 + ca.exp(-K*(f - f_min))) # Effective thrust
            F_eff = stage.prop.F_SL*f
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

            ### ODE FUNC AND INTEGRATOR ###
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

            N = self.N[k]
            for i in range(N):
                # Do gap closing ig?
                G.append(X[k][i+1] - F_int(X[k][i], U[k][i], T[k][i]/N))
                G.append(T[k][i+1] - T[k][i])
                E += (nx + 1)*[True]

                if i == 0 and k == 0: # Has to be here to give fatrop correct C, D matrix structure
                    # Add in initial constraints
                    gb_0 = self.x0.get_ge(X[0][0], T, self)
                    G += gb_0['g']
                    E += gb_0['e']
                else:
                    # path constraints, make sure vehicle does not go below planet ig
                    G.append(ca.sumsqr(ca.vertcat(X[k][i][1:4])) - self.body.r_0**2)
                    E.append(False)
                    # pass

            if k+1 < self.nstages:
                # Continuity between stages
                G.append(X[k+1][0][0] - X[k][-1][0] - (self.stages[k+1].m_0 - self.stages[k].m_f)) # mass
                G.append(X[k+1][0][1:7] - X[k][-1][1:7]) # pos/vel/ctrl
                E += (nx)*[True]

                # path constraints, make sure vehicle does not go below planet ig
                G.append(ca.sumsqr(ca.vertcat(X[k][-1][1:4])) - self.body.r_0**2)
                E.append(False)

        # Add in final constraints
        gb_f = self.xf.get_ge(X[-1][-1], T, self)
        G += gb_f['g']
        E += gb_f['e']

        # Optimization function
        opt_func = (self.stages[-1].m_0 - X[-1][-1][0])/(self.stages[-1].m_0 - self.stages[-1].m_f)

        # Create solver
        nlp = {'x': ca.vertcat(*V), 'f': opt_func, 'g': ca.vertcat(*G)}

        if self.extra_opts.get('solver') == 'ipopt':
            ipopt_opts = {
                'expand': self.config.integration_method == 'RK4',
                'ipopt.nlp_scaling_method': 'none',
                'ipopt.tol': self.config.solver_tol
            }
            nlpsolver = ca.nlpsol(
                'nlpsolver', 'ipopt', nlp, ipopt_opts
                )
        else:
            fatrop_opts = {
                'expand': self.config.integration_method == 'RK4',
                'fatrop': {"mu_init": 0.1},
                'structure_detection': 'auto',
                'debug': True,
                'equality': E,
            }
            nlpsolver = ca.nlpsol(
                'nlpsolver', 'fatrop', nlp, fatrop_opts
            )
        self.nlpsolver = nlpsolver
        self.nlp_creation_time = time.time() - start_time

    def solve_nlp_2(self) -> None:
        start_time = time.time()
        if self.nlpsolver is None:
            raise Exception('NLP Solver must be created with create_nlp()')
        elif not self.initialized:
            raise Exception('Solver must be initialized with a guess solution.')
        
        # maybe move to solver properties later?
        nx = 7 
        nu = 3
        x0, lbx, ubx, lbg, ubg = [], [], [], [], []
        # create x0
        # for k in range(self.nstages):
        #     for i in range(self.N[k]):
        #         x0 += self.sols[-1][k].X[i] + [self.sols[-1][k].t[-1]] + self.sols[-1][k].U[i]
        #     x0 += self.sols[-1][k].X[-1] + [self.sols[-1][k].t[-1]]
        for k in range(self.nstages):
            for i in range(self.N[k]):
                x0 += self.sols[-1][k].X[i][0:7] + [self.sols[-1][k].t[-1]] + self.sols[-1][k].X[i][7:10]
            x0 += self.sols[-1][k].X[-1][0:7] + [self.sols[-1][k].t[-1]]

        # create lbg ubg
        for k in range(self.nstages):
            for i in range(self.N[k]):
                # gap closing, physics and time have to match
                lbg += (nx+1)*[0]
                ubg += (nx+1)*[0]

                if i == 0 and k == 0:
                    # initial constraints lbg ubg
                    gb_0 = self.x0.get_gb(self)
                    lbg += gb_0['lbg']
                    ubg += gb_0['ubg']
                else:
                    # path constraints
                    lbg.append(0)
                    ubg.append(109225)
                    # pass

            if k+1 < self.nstages:
                # stage continuity
                lbg += nx*[0]
                ubg += nx*[0]

                # path constraints
                lbg.append(0)
                ubg.append(109225)


        # final constraint lbg ubg
        gb_f = self.xf.get_gb(self)
        lbg += gb_f['lbg']
        ubg += gb_f['ubg']

        # create lbx ubx
        # free states, may adjust later
        ubx_free = 6*[ca.inf]
        lbx_free = 6*[-ca.inf]
        ubu_free = [1, ca.pi, ca.pi/2]
        lbu_free = [0, -ca.pi, -ca.pi/2]
        for k, stage in enumerate(self.stages):
            m_0 = stage.m_0
            m_f = stage.m_f
            T_min = self.T_min[k]
            T_max = self.T_max[k]

            if k+1 == 1:
                # first node first stage
                xb_0 = self.x0.get_xb(self)
                lbx += [m_0] + xb_0['lbx'][0:6] + [T_min] + lbu_free
                ubx += [m_0] + xb_0['ubx'][0:6] + [T_max] + ubu_free
            else:
                # first node other stages
                lbx += [m_0] + lbx_free + [T_min] + lbu_free
                ubx += [m_0] + ubx_free + [T_max] + ubu_free

            lbx += (self.N[k]-1)*([m_f] + lbx_free + [T_min] + lbu_free)
            ubx += (self.N[k]-1)*([m_0] + ubx_free + [T_max] + ubu_free)

            if k+1 == self.nstages:
                # last node last stage
                xb_f = self.xf.get_xb(self)
                lbx += [m_f] + xb_f['lbx'][0:6] + [T_min]
                ubx += [m_0] + xb_f['ubx'][0:6] + [T_max]
            else:
                # last node other stages
                lbx += [m_f] + lbx_free + [T_min]
                ubx += [m_f] + ubx_free + [T_max]

        # SOLVE #
        result = self.nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # PARSE RESULTS #
        V_res = np.array(result['x']).flatten()
        N_arr = np.array(self.N)
        blocks = np.concatenate(([0], np.cumsum((N_arr+1)*(nx+1) + N_arr*nu)))
        intersols = []
        for k in range(self.nstages):
            N = self.N[k]
            Vk = V_res[blocks[k] : blocks[k+1]]
            Xk = np.array([Vk[(nx+nu+1)*i : (nx+nu+1)*(i+1) - nu - 1] for i in range(N+1)])
            Tk = np.array([Vk[(nx+nu+1)*i + nx] for i in range(N+1)])
            Uk = np.array([Vk[(nx+nu+1)*i + nx + 1 : (nx+nu+1)*(i+1)] for i in range(N)])
            t_res = np.linspace(0, Tk[-1], N + 1)
            sol = Solution(X=Xk.tolist(),
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
