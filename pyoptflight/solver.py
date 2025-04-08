# from __future__ import annotations
from .functions import *
from .setup import *
from .boundary_objects import *
from typing import List, Callable, Dict, Union
from dataclasses import dataclass, field
import time
from .physics import StagePhysics

@dataclass(repr=False)
class SolverContext(AutoRepr):
    """Context object for the solver. Designed to be passed around to functions."""
    body: "Body"                                  # Immutable
    stages: list["Stage"]                         # Immutable
    config: "SolverConfig"                        # Immutable
    nstages: int = 0                              # Immutable
    N: list[int] = field(default_factory=list)    # Immutable
    T_init: list[float] = field(default_factory=list)  # Immutable
    T_min: list[float] = field(default_factory=list)   # Immutable
    T_max: list[float] = field(default_factory=list)   # Immutable
    T_sum: float = 0.0                                # Mutable
    Nx: int = 0                                   # Immutable
    Nu: int = 0                                   # Immutable
    use_atm: bool = False                         # Immutable
    delta: float = 0.01                           # Immutable

    def __post_init__(self):
        # Assign T_init, T_min, T_max lists using global default in config or stage value
        attributes = ['N', 'T_init', 'T_min', 'T_max']
        for attr in attributes:
            setattr(self, attr, [
                getattr(self.config, attr) if getattr(stage, attr) is None else getattr(stage, attr)
                for stage in self.stages
            ])

        self.nstages = len(self.stages)
        self.Nu = sum(self.N)
        self.Nx = self.Nu + self.nstages
        self.T_sum = sum(self.T_init)
        self.use_atm = False if self.body.atm is None else True

@dataclass(repr=False)
class StageSolution(AutoRepr):
    """Obejct to store the soltuion(s)"""
    X: list[List] ### self.config.N+1 x 7 List of Lists of state solutions
    U: list[List] ### self.config.N x 3 List of Lists of control solutions
    stage: int ### Stage number solution belongs to
    T: float ### Time for this stage will replace T_init

class FlightSolution(AutoRepr):
    """Data computed after solving. Flight stats."""
    @dataclass(repr=False)
    class TimeSeriesData(AutoRepr):
        data: np.ndarray
        times: np.ndarray
        nodes: np.ndarray
    
    def __init__(self, sols: List[StageSolution], 
                 context: "SolverContext",
                 constraints: List[ConstraintSet]):
        
        def stack_cont(arrays: List[np.ndarray]) -> np.ndarray:
            result = arrays[0]
            for i in range(1, len(arrays)):
                result = np.concatenate((result, arrays[i][1:]))
            return result
        
        def compute_phys(func_name: str, 
                         args: List, 
                         stage_bounds: List, 
                         times: np.ndarray,
                         nodes: np.ndarray,
                         output_size=0):
            if output_size == 0:
                result = np.zeros((stage_bounds[-1]))
            else:
                result = np.zeros((stage_bounds[-1], output_size))
            for k, stage in enumerate(context.stages):
                if constraints[k].f_min.enabled and constraints[k].f_min.value is not None:
                    f_min = constraints[k].f_min.value
                else:
                    f_min = 0
                physics = StagePhysics(context, stage, f_min)
                func = getattr(physics, func_name)
                for i in range(stage_bounds[k], stage_bounds[k+1]):
                    if output_size > 0:
                        result[i] = np.array(func(*[arg[i] for arg in args])).flatten()
                    else:
                        result[i] = func(*[arg[i] for arg in args])
            return self.TimeSeriesData(result, times, nodes)


        cum_T = np.concatenate(([0], np.cumsum([sol.T for sol in sols])))
        cum_N = np.concatenate(([0], np.cumsum(context.N))) # stage boundaries by index
        t_segs = [np.linspace(cum_T[i], cum_T[i+1], context.N[i]+1) for i in range(context.nstages)]
        N_segs = [np.arange(cum_N[i], cum_N[i+1]+1) for i in range(context.nstages)]
        t_cont = stack_cont(t_segs)
        N_cont = stack_cont(N_segs)
        t_disc = np.concatenate(t_segs)
        N_disc = np.concatenate(N_segs)
        X_cont = stack_cont([sol.X for sol in sols])
        X_disc = np.concatenate([sol.X for sol in sols])
        U_cont = np.concatenate([sol.U for sol in sols])

        self.stage_num = self.TimeSeriesData(np.concatenate([sols[i].stage*np.ones(context.N[i]+1) for i in range(context.nstages)]), t_disc, N_disc) # stage number
        self.pos = self.TimeSeriesData(X_cont[:, 1:4], t_cont, N_cont) # px, py, pz
        self.vel = self.TimeSeriesData(X_cont[:, 4:7], t_cont, N_cont) # vx, vy, vz
        self.mass = self.TimeSeriesData(X_disc[:, 0], t_disc, N_disc)  # m
        self.f = self.TimeSeriesData(U_cont[:, 0], t_cont[:-1], N_cont[:-1]) # f
        self.psi = self.TimeSeriesData(U_cont[:, 1], t_cont[:-1], N_cont[:-1]) # psi
        self.theta = self.TimeSeriesData(U_cont[:, 2], t_cont[:-1], N_cont[:-1]) # theta1

        cum_N_cont = np.copy(cum_N)
        cum_N_cont[-1] += 1
        self.h     = compute_phys('h',         [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont)
        self.f_eff = compute_phys('f_eff',     [U_cont[:, 0]],        cum_N,      t_cont[:-1], N_cont[:-1])
        self.F_max = compute_phys('F_max',     [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont)
        self.Isp   = compute_phys('Isp',       [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont)
        self.rho   = compute_phys('rho',       [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont)
        self.g     = compute_phys('g',         [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont, 3)
        self.wind  = compute_phys('wind',      [*X_cont[:, 1:4].T],   cum_N_cont, t_cont,      N_cont, 3)
        self.v_rel = compute_phys('v_rel',     [*X_cont[:, 1:7].T],   cum_N_cont, t_cont,      N_cont, 3)
        self.ebx   = compute_phys('ebx',       [*U_cont[:, 1:3].T],   cum_N,      t_cont[:-1], N_cont[:-1], 3)
        self.eby   = compute_phys('eby',       [*U_cont[:, 1:3].T],   cum_N,      t_cont[:-1], N_cont[:-1], 3)
        self.ebz   = compute_phys('ebz',       [*U_cont[:, 1:3].T],   cum_N,      t_cont[:-1], N_cont[:-1], 3)
        self.q     = compute_phys('q',         [X_cont],              cum_N_cont, t_cont,      N_cont)
        self.alpha = compute_phys('cos_alpha', [X_cont[:-1], U_cont], cum_N,      t_cont[:-1], N_cont[:-1])       
        self.alpha.data = np.arccos(np.minimum(1, self.alpha.data)) # convert to angle
        # tau?
        # body rates?

class Solver(AutoRepr):
    def __init__(self, 
                 body: "Body", 
                 stages: List["Stage"], 
                 config: "SolverConfig", 
                 x0: BoundaryObj, 
                 xf: BoundaryObj):
        
        self.context = SolverContext(body, stages, config)
        self.x0 = x0 ### Starting point (obj)
        self.xf = xf ### Ending point (obj)

        self.stage_sols: List[List[StageSolution]] = []
        self.flight_sols: List[FlightSolution] = []

        # Can pass sols to pre-initialize, but defaults to uninitialized. TODO: May remove. 
        self.initialized = False
        self.status = None
        self.success = True
        self.runtime = 0
        self.nlp_creation_time = 0
        self.iter_count = 0
        self.nsolves = 0

        # Select constraints
        self.constraints = [None]*self.context.nstages
        self.merge_constraints(force_source=config.force_constraints)

    def stats(self) -> Dict:
        """Returns basic stats of overall solve"""
        return {'status': self.status,
                'success': self.success,
                'runtime': self.runtime,
                'iter_count': self.iter_count,
                'T': self.context.T_sum,
                'nsolves': self.nsolves,
                'nlp_creation_time': self.nlp_creation_time,
                'timestep_sizes': [self.context.T_init[i]/self.context.N[i] for i in range(self.context.nstages)],
                'final_mass': self.stage_sols[-1][-1].X[-1][0]}
    
    def update_stats(
        self,
        sols: List[StageSolution],
        status: Optional[str] = None,
        success: Optional[bool] = None,
        iter_count: Optional[int] = None,
        runtime: Optional[float] = None,
    ) -> None:
        self.stage_sols.append(sols)
        self.flight_sols.append(FlightSolution(sols, self.context, self.constraints))

        self.status = status if status is not None else self.status
        self.success = success if success is not None else self.success
        self.iter_count = iter_count if iter_count is not None else self.iter_count
        self.runtime = runtime if runtime is not None else self.runtime

        self.context.T_sum = sum(sol.T for sol in sols)
        self.nsolves = len(self.stage_sols)

    def initialize_from_func(self, init_func: Callable, opts: Dict) -> None:
        start_time = time.time()
        res = init_func(self.context, self.x0, self.xf, opts)
        runtime = time.time() - start_time

        self.update_stats(
            sols=res['sols'],
            status=res['status'],
            success=res['success'],
            iter_count=res['iter_count'],
            runtime=runtime,
        )
        self.initialized = self.success

    def initialize_from_sols(self, sols: List[StageSolution]) -> None:
        self.update_stats(
            sols=sols,
            status=None,
            success=True,
            iter_count=0,
            runtime=0
        )
        self.initialized = True

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
                global_cs=self.context.config.global_constraints,
                stage_cs=stage.constraints,
                force_source=force_source
            )
            for stage in self.context.stages
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

        ### symbolic state and control vectors ###
        x = ca.SX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
        u = ca.SX.sym('[f, psi, theta]', 3, 1)

        nx = x.size1() # Number of states (10)
        nu = u.size1() # number of control vars (3)

        V = ca.MX.sym('V', sum(self.context.N)*(nx + nu) + self.context.nstages + nx)
        T = V[0:self.context.nstages]
        T_sum = ca.sum1(T)
        X = []
        U = []
        G = []
        
        # all blocks start after nstages number of T's
        # each block has N nx and nu with an additional sometimes overlapping nx
        block_bounds = np.concatenate(([0], (nx+nu)*np.cumsum(self.context.N))) + self.context.nstages
        for k in range(self.context.nstages):
            N = self.context.N[k]
            Vk = V[block_bounds[k] : block_bounds[k+1] + nx]
            Uk = [Vk[(nx+nu)*i + nx : (nx+nu)*(i+1)] for i in range(N)]
            Xk = [Vk[(nx+nu)*i: (nx+nu)*(i+1) - nu] for i in range(N + 1)]
            U.append(Uk)
            X.append(Xk)

        for k, stage in enumerate(self.context.stages):
            ###################
            ### PHYSICS/ODE ###
            ###################

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

            stage_physics = StagePhysics(self.context, stage, f_min)
            ode = stage_physics.ode(x, u)

            F_ode = ca.Function('F_ode', [x, u], [ode])
            # All integrators need x, u, dt (symbolics) and should return x_next
            dt = ca.SX.sym("dt")
            if self.context.config.integration_method == 'RK4': # Implement more int methods later RK4
                k1 = F_ode(x, u)
                k2 = F_ode(x + dt/2 * k1, u)
                k3 = F_ode(x + dt/2 * k2, u)
                k4 = F_ode(x + dt * k3, u)
                x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                F_int = ca.Function('F_int', [x, u, dt], [x_next])
            elif self.context.config.integration_method == 'cvodes':
                dae = {'x': x, 'u':u, 'p': dt, 'ode': dt*F_ode(x, u)}
                int_opts = {'nonlinear_solver_iteration': 'functional'}
                I = ca.integrator('I', 'cvodes', dae, 0.0, 1.0, int_opts)
                x_mx = ca.MX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
                u_mx = ca.MX.sym('[f, psi, theta]', 3, 1)
                dt_mx = ca.MX.sym('dt_mx')
                F_int = ca.Function('F_int', [x_mx, u_mx, dt_mx], [I(x0=x_mx, u=u_mx, p=dt_mx)['xf']])
            else:
                raise NotImplementedError(f'{self.context.config.integration_method} is not an implmented integrator.')
            
            ##############################
            ### CONSTRAINT APPLICATION ###
            ##############################

            N = self.context.N[k]
            for i in range(N): # iterate through N stage nodes
                # gap closing
                if i+1 == N and k+1 < self.context.nstages: # if last node of a booster stage
                    # stage i+1 mass plus stage i empty mass must equal integration of stage i mass flow
                    m_e = self.context.stages[k].m_f - self.context.stages[k+1].m_0 # stage empty mass
                    G.append(X[k][i+1][0] + m_e - F_int(X[k][i], U[k][i], T[k]/N)[0])
                    G.append(X[k][i+1][1:] - F_int(X[k][i], U[k][i], T[k]/N)[1:])

                else: # all other nodes
                    G.append(X[k][i+1] - F_int(X[k][i], U[k][i], T[k]/N))

                if i == 0 and k == 0: # if first node and first stage
                    # initial constraint placed here for sparcity
                    ge_0 = self.x0.get_ge(X[0][0], U[0][0], T_sum, self.context)
                    G += ge_0['g']
                else: # all other nodes
                    # add path constraints
                    # do not let path dip below planet radius
                    G.append(ca.sumsqr(ca.vertcat(X[k][i][1:4])) - self.context.body.r_0**2)

                    if f_min_constr.enabled and f_min_constr.value is not None:
                        G.append((U[k][i][0] - f_min_constr.value + self.context.delta)*(U[k][i][0] - f_min_constr.value))
                    # constraints currently do not consider stage interface!
                    if q_constr.enabled and q_constr.value is not None: # max q
                        G.append(stage_physics.q(X[k][i]))
                    if alpha_constr.enabled and alpha_constr.value is not None: # max AoA
                        G.append(stage_physics.cos_alpha(X[k][i], U[k][i]))
                
                # rate constraints (temporary)
                if i+1 < N: # if not last node
                    dt = T[k]/N
                    if tau_constr.enabled and tau_constr.value is not None:
                        G.append((U[k][i+1][0]-U[k][i][0])/dt)
                    if body_rate_y_constr.enabled and body_rate_y_constr.value is not None:
                        G.append((U[k][i+1][1]-U[k][i][1])/dt*ca.cos(U[k][i][2]))
                    if body_rate_z_constr.enabled and body_rate_z_constr.value is not None:
                        G.append((U[k][i+1][2]-U[k][i][2])/dt)

        # final constraint placed here for sparcity
        ge_f = self.xf.get_ge(X[-1][-1], U[-1][-1], T_sum, self.context)
        G += ge_f['g']

        # Optimization function
        opt_func = (self.context.stages[-1].m_0 - X[-1][-1][0])/(self.context.stages[-1].m_0 - self.context.stages[-1].m_f)

        # Create solver
        nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}
        ipopt_opts = {
            'expand': self.context.config.integration_method == 'RK4',
            'ipopt.nlp_scaling_method': 'none',
            'ipopt.tol': self.context.config.solver_tol,
            'ipopt.max_iter': self.context.config.max_iter
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
        x0 += [sol.T for sol in self.stage_sols[-1]]
        for k in range(self.context.nstages):
            for i in range(self.context.N[k]):
                x0 += self.stage_sols[-1][k].X[i] + self.stage_sols[-1][k].U[i]
        x0 += self.stage_sols[-1][-1].X[-1]

        # create lbg ubg
        for k in range(self.context.nstages):
            f_min_constr = self.constraints[k].f_min
            q_constr = self.constraints[k].max_q
            alpha_constr = self.constraints[k].max_alpha
            body_rate_y_constr = self.constraints[k].max_body_rate_y
            body_rate_z_constr = self.constraints[k].max_body_rate_z
            tau_constr = self.constraints[k].max_tau
            for i in range(self.context.N[k]):
                # gap closing:
                lbg += nx*[0]
                ubg += nx*[0]

                if i == 0 and k == 0:
                    # initial constraints
                    gb_0 = self.x0.get_gb(self.context)
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
                        lbg.append(np.cos(alpha_constr.value))
                        ubg.append(1)
                if i+1 < self.context.N[k]: # if not last node
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
        gb_f = self.xf.get_gb(self.context)
        lbg += gb_f['lbg']
        ubg += gb_f['ubg']

        # create lbx ubx
        free_bound = FreeBound()
        xb_free = free_bound.get_xb(self.context)
        ub_free = free_bound.get_ub(self.context)
        lbx += self.context.T_min
        ubx += self.context.T_max
        for k, stage in enumerate(self.context.stages):
            N = self.context.N[k]
            m_0 = stage.m_0
            m_f = stage.m_f
            f_min_constr = self.constraints[k].f_min
            if f_min_constr.enabled and f_min_constr.value is not None:
                lbf = max(f_min_constr.value - self.context.delta, 0)
            else:
                lbf = 0
            
            # bounds on first node
            if k == 0:
                xb_0 = self.x0.get_xb(self.context)
                ub_0 = self.x0.get_ub(self.context)
                lbx += [m_0] + xb_0['lbx'] + [lbf] + ub_0['lbu']
                ubx += [m_0] + xb_0['ubx'] + [1]   + ub_0['ubu']
            else: # stage interface nodes are here vvv
                lbx += [m_0] + xb_free['lbx'] + [lbf] + ub_free['lbu']
                ubx += [m_0] + xb_free['ubx'] + [1]   + ub_free['ubu']

            # bounds on next N-1 nodes (N total at this point)
            if k + 1 < self.context.nstages: # if not last stage
                lbx += (N-1)*([m_f] + xb_free['lbx'] + [lbf] + ub_free['lbu'])
                ubx += (N-1)*([m_0] + xb_free['ubx'] + [1]   + ub_free['ubu'])
            else: # Last stage gets extra xb and special ub 
                lbx += (N-2)*([m_f] + xb_free['lbx'] + [lbf] + ub_free['lbu'])
                ubx += (N-2)*([m_0] + xb_free['ubx'] + [1]   + ub_free['ubu'])

                # final constraint
                xb_f = self.xf.get_xb(self.context)
                ub_f = self.xf.get_ub(self.context)
                lbx += [m_f] + xb_free['lbx'] + [lbf] + ub_f['lbu'] + [m_f] + xb_f['lbx']
                ubx += [m_0] + xb_free['ubx'] + [1]   + ub_f['ubu'] + [m_0] + xb_f['ubx']

        # SOLVE #
        result = self.nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # # PARSE RESULTS #
        intersols = []
        V_res = np.array(result['x']).flatten()
        block_bounds = np.concatenate(([0], (nx+nu)*np.cumsum(self.context.N))) + self.context.nstages
        for k in range(self.context.nstages):
            N = self.context.N[k]
            Vk = V_res[block_bounds[k] : block_bounds[k+1] + nx]
            Uk = np.array([Vk[(nx+nu)*i + nx : (nx+nu)*(i+1)] for i in range(N)])
            Xk = np.array([Vk[(nx+nu)*i: (nx+nu)*(i+1) - nu] for i in range(N + 1)])

            # add back in structural mass at end of stage
            if k + 1 < self.context.nstages:
                m_e = self.context.stages[k].m_f - self.context.stages[k+1].m_0 # stage empty mass
                Xk[-1][0] += m_e

            sol = StageSolution(
                X=Xk.tolist(),
                U=Uk.tolist(),
                stage=k+1,
                T=V_res[k],
                )
            intersols.append(sol)

        status = self.nlpsolver.stats()['return_status']
        success = status == 'Solve_Succeeded'
        iter_count = self.nlpsolver.stats()['iter_count']
        runtime = time.time() - start_time

        self.update_stats(
            sols=intersols,
            status=status,
            success=success,
            iter_count=iter_count,
            runtime=runtime,
            )