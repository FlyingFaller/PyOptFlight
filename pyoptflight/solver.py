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
    body   : "Body"                                    # Immutable
    stages : list["Stage"]                             # Immutable
    config : "SolverConfig"                            # Immutable
    nstages: int = 0                                   # Immutable
    N      : list[int] = field(default_factory=list)   # Immutable
    T_init : list[float] = field(default_factory=list) # Immutable
    T_min  : list[float] = field(default_factory=list) # Immutable
    T_max  : list[float] = field(default_factory=list) # Immutable
    T_sum  : float = 0.0                               # Mutable
    Nx     : int = 0                                   # Immutable
    Nu     : int = 0                                   # Immutable
    use_atm: bool = False                              # Immutable
    delta  : float = 0.01                              # Immutable

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
    X    : list[List] ### self.config.N+1 x 7 List of Lists of state solutions
    U    : list[List] ### self.config.N x 3 List of Lists of control solutions
    stage: int        ### Stage number solution belongs to
    T    : float      ### Time for this stage will replace T_init

class FlightSolution(AutoRepr):
    """Data computed after solving. Flight stats."""
    @dataclass(repr=False)
    class TimeSeriesData(AutoRepr):
        data: np.ndarray = np.empty(0)
        times: np.ndarray = np.empty(0)
        nodes: np.ndarray = np.empty(0, dtype=int)
        constraint: list = field(default_factory=list)
        
        def update(self, data: float|np.ndarray|list, time: float, node: int, constraint: float|None = None) -> None:
            data_list: list = self.data.tolist()
            data_list.append(data)
            self.data = np.array(data_list)
            self.times = np.concatenate((self.times, [time]))
            self.nodes = np.concatenate((self.nodes, [node]))
            self.constraint.append(constraint)
            
    def __init__(self, sols: List[StageSolution], 
                 context: "SolverContext",
                 constraints: List[ConstraintSet]):
        
        def stack(arrays) -> np.ndarray:
            result = np.array(arrays[0])
            for i in range(1, len(arrays)):
                result = np.concatenate((result, arrays[i][1:]))
            return result
        
        self.stage_num   = self.TimeSeriesData() # kN+k
        self.pos         = self.TimeSeriesData() # kN+1
        self.vel         = self.TimeSeriesData() # kN+1
        self.mass        = self.TimeSeriesData() # kN+k
        self.f           = self.TimeSeriesData() # kN
        self.psi         = self.TimeSeriesData() # kN
        self.theta       = self.TimeSeriesData() # kN
        self.h           = self.TimeSeriesData() # kN+1
        self.f_eff       = self.TimeSeriesData() # kN
        self.F_max       = self.TimeSeriesData() # kN
        self.Isp         = self.TimeSeriesData() # kN+1
        self.rho         = self.TimeSeriesData() # kN+1
        self.g           = self.TimeSeriesData() # kN+1
        self.wind        = self.TimeSeriesData() # kN+1
        self.ebx         = self.TimeSeriesData() # kN
        self.eby         = self.TimeSeriesData() # kN
        self.ebz         = self.TimeSeriesData() # kN
        self.q           = self.TimeSeriesData() # kN+1
        self.alpha       = self.TimeSeriesData() # kN
        self.tau         = self.TimeSeriesData() # kN-1
        self.body_rate_y = self.TimeSeriesData() # kN-1
        self.body_rate_z = self.TimeSeriesData() # kN-1

        N  = context.N
        cum_T = np.concatenate(([0], np.cumsum([sol.T for sol in sols])))
        cum_N = np.concatenate(([0], np.cumsum(N))) # stage boundaries by index
        sum_N = np.sum(N) # total number of nodes kN
        nstages = context.nstages
        stages = context.stages
        cs_list = constraints
        t_list = stack([np.linspace(cum_T[i], cum_T[i+1], N[i]+1) for i in range(nstages)])
        N_list = stack([np.arange(cum_N[i], cum_N[i+1]+1) for i in range(nstages)])
        X = stack([sol.X for sol in sols])
        U = np.concatenate([sol.U for sol in sols])

        physics: List[StagePhysics] = []
        for k in range(nstages):
            f_min_constr = cs_list[k].f_min
            if f_min_constr.enabled and f_min_constr.value is not None:
                f_min = f_min_constr.value
            else:
                f_min = 0
            physics.append(StagePhysics(context, stages[k], f_min))

        for i in range(sum_N+1):
            k = np.searchsorted(cum_N[:-1], i, side='right') - 1 # 'current' stage index, this breaks down at stage interfaces
            stage_interface = i in cum_N[1:-1] # True if i is a stage interface node
            first_node = i+1 == 1 # True if first node of loop 
            last_node = i+1 == sum_N+1 # True if last node of loop
            penult_node = i+1 == sum_N # True if second to last node of loop
            t = t_list[i]
            n = N_list[i]
            sol = sols[k]

            # constraints
            max_cs = ConstraintSet.choose_max(cs_list[k], cs_list[k-1]) if stage_interface else cs_list[k]
            f_min = cs_list[k].f_min.value if cs_list[k].f_min.enabled else None
            max_alpha = cs_list[k].max_alpha.value if cs_list[k].max_alpha.enabled else None
            max_tau = max_cs.max_tau.value if max_cs.max_tau.enabled else None
            max_body_rate_y = max_cs.max_body_rate_y.value if max_cs.max_body_rate_y.enabled else None
            max_body_rate_z = max_cs.max_body_rate_z.value if max_cs.max_body_rate_z.enabled else None
            max_q = max_cs.max_q.value if max_cs.max_q.enabled else None
            f_min = f_min if not (first_node or last_node or penult_node) else None
            max_q = max_q if not (first_node or last_node) else None
            max_alpha = max_alpha if not (first_node or last_node or penult_node) else None

            # kN+1 nodes
            self.pos.update(X[i][1:4], t, n)
            self.vel.update(X[i][4:7], t, n)
            self.h.update(float(physics[k].h(*X[i][1:4])), t, n)
            self.Isp.update(float(physics[k].Isp(*X[i][1:4])), t, n)
            self.rho.update(float(physics[k].rho(*X[i][1:4])), t, n)
            self.g.update(np.array(physics[k].g(*X[i][1:4])).flatten(), t, n)
            self.wind.update(np.array(physics[k].wind(*X[i][1:4])).flatten(), t, n)
            self.F_max.update(float(physics[k].F_max(*X[i][1:4])), t, n)
            self.q.update(float(physics[k].q(X[i])), t, n, max_q)

            # kN+k nodes
            self.mass.update(X[i][0], t, n)
            if stage_interface:
                self.stage_num.update(k, t, n)
                self.mass.update(sols[k].X[0][0], t, n)
            self.stage_num.update(k+1, t, n)

            # kN nodes
            if not last_node: # controls
                self.f.update(U[i][0], t, n, f_min)
                self.psi.update(U[i][1], t, n)
                self.theta.update(U[i][2], t, n)
                self.f_eff.update(float(physics[k].f_eff(U[i][0])), t, n, f_min)
                self.ebx.update(np.array(physics[k].ebx(*U[i][1:])).flatten(), t, n)
                self.eby.update(np.array(physics[k].eby(*U[i][1:])).flatten(), t, n)
                self.ebz.update(np.array(physics[k].ebz(*U[i][1:])).flatten(), t, n)
                self.alpha.update(float(np.arccos(min(1, physics[k].cos_alpha(X[i], U[i])))), t, n, max_alpha)
            
            # kN-1
            if not first_node and not last_node: # control rates
                dt = sol.T/N[k]
                dudt = (np.array(U[i]) - np.array(U[i-1]))/dt
                self.tau.update(dudt[0], t, n, max_tau)
                self.body_rate_y.update(dudt[1]*np.cos((U[i][2] + U[i-1][2])/2), t, n, max_body_rate_y)
                self.body_rate_z.update(dudt[2], t, n, max_body_rate_z)

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

        # some helpful variables
        N  = self.context.N
        cum_N = np.concatenate(([0], np.cumsum(N))) # stage boundaries by index
        sum_N = np.sum(N) # total number of nodes kN
        nstages = self.context.nstages
        stages = self.context.stages
        cs_list = self.constraints

        ### symbolic state and control vectors ###
        x = ca.SX.sym('[m, px, py, pz, vx, vy, vz]', 7, 1)
        u = ca.SX.sym('[f, psi, theta]', 3, 1)

        nx = x.size1() # Number of states (10)
        nu = u.size1() # number of control vars (3)

        V = ca.MX.sym('V', nstages + sum_N*(nx + nu) + nx)
        T = V[0:nstages]
        T_sum = ca.sum1(T)
        X = [V[nstages + i*(nx + nu): nstages + (i+1)*(nx + nu) - nu] for i in range(sum_N+1)] # select N+1 Xs
        U = [V[nstages + i*(nx + nu) + nx: nstages + (i+1)*(nx + nu)] for i in range(sum_N)] # select N Us
        G = [] # leave constraints to be filled out later
        
        # setup physics and itegrators for each stage
        physics: List[StagePhysics] = []
        integrators = []
        for k in range(nstages):
            f_min_constr = cs_list[k].f_min
            if f_min_constr.enabled and f_min_constr.value is not None:
                f_min = f_min_constr.value
            else:
                f_min = 0
            stage_physics = StagePhysics(self.context, stages[k], f_min)
            physics.append(stage_physics)
            ode = stage_physics.ode(x, u)
            F_ode = ca.Function('F_ode', [x, u], [ode])
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
            integrators.append(F_int)

        for i in range(0, sum_N): # loop through 1st to kNth node, exluding kN+1th node
            k = np.searchsorted(cum_N[:-1], i, side='right') - 1 # 'current' stage index, this breaks down at stage interfaces
            stage_interface = i in cum_N[1:-1] # True if i is a stage interface node
            first_loop_node = i+1 == 1 # True if first node of loop 
            last_loop_node = i+1 == sum_N # True if last node of loop

            # gap closing
            if i + 1 in cum_N[1:-1]: # if penultimate node in a stage
                m_e = stages[k].m_f - stages[k+1].m_0 # stage empty mass
                G.append(X[i+1][0] + m_e - integrators[k](X[i], U[i], T[k]/N[k])[0])
                G.append(X[i+1][1:] - integrators[k](X[i], U[i], T[k]/N[k])[1:])
            else:
                G.append(X[i+1] - integrators[k](X[i], U[i], T[k]/N[k]))

            # radial constraint on interior nodes
            if first_loop_node: # first node gets boundary condition
                ge_0 = self.x0.get_ge(X[i], U[i], T_sum, self.context)
                G += ge_0['g']
            else: # interior points
                G.append(ca.sumsqr(ca.vertcat(X[i][1:4])) - self.context.body.r_0**2)

                # choose current constraints and deal with stage interfaces
                if stage_interface:
                    max_cs = ConstraintSet.choose_max(cs_list[k-1], cs_list[k])
                else:
                    max_cs = cs_list[k]

                # rate and q constraints
                dt = T[k]/N[k]
                if max_cs.max_tau.enabled and max_cs.max_tau.value is not None:
                    G.append((U[i][0]-U[i-1][0])/dt)
                if max_cs.max_body_rate_y.enabled and max_cs.max_body_rate_y.value is not None:
                    G.append((U[i][1]-U[i-1][1])/dt*ca.cos((U[i][2] + U[i-1][2])/2))
                if max_cs.max_body_rate_z.enabled and max_cs.max_body_rate_z.value is not None:
                    G.append((U[i][2]-U[i-1][2])/dt)
                if max_cs.max_q.enabled and max_cs.max_q.value is not None: # max q
                    G.append(physics[k].q(X[i]))

            if not first_loop_node and not last_loop_node: # control node constraints
                # f_min and alpha constraints
                if cs_list[k].f_min.enabled and cs_list[k].f_min.value is not None:
                    G.append((U[i][0] - cs_list[k].f_min.value + self.context.delta)*(U[i][0] - cs_list[k].f_min.value))
                if cs_list[k].max_alpha.enabled and cs_list[k].max_alpha.value is not None:
                    G.append(physics[k].cos_alpha(X[i], U[i]))

        ge_f = self.xf.get_ge(X[-1], U[-1], T_sum, self.context)
        G += ge_f['g']

        opt_func = (stages[-1].m_0 - X[-1][0])/(stages[-1].m_0 - stages[-1].m_f)

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

        # some helpful variables
        N  = self.context.N
        cum_N = np.concatenate(([0], np.cumsum(N))) # stage boundaries by index
        sum_N = np.sum(N) # total number of nodes kN
        nstages = self.context.nstages
        stages = self.context.stages
        cs_list = self.constraints

        # create x0
        x0 += [sol.T for sol in self.stage_sols[-1]]
        for k in range(nstages):
            for i in range(self.context.N[k]):
                x0 += self.stage_sols[-1][k].X[i] + self.stage_sols[-1][k].U[i]
        x0 += self.stage_sols[-1][-1].X[-1]

        # create start of lbx ubx and free bound
        free_bound = FreeBound()
        xb_free = free_bound.get_xb(self.context)
        ub_free = free_bound.get_ub(self.context)
        lbx += self.context.T_min
        ubx += self.context.T_max

        for i in range(0, sum_N): # loop through 1st to kNth node, exluding kN+1th node
            k = np.searchsorted(cum_N[:-1], i, side='right') - 1 # 'current' stage index, this breaks down at stage interfaces
            stage_interface = i in cum_N[1:-1] # True if i is a stage interface node
            first_loop_node = i+1 == 1 # True if first node of loop 
            last_loop_node = i+1 == sum_N # True if last node of loop

            # mass for current stage
            m_0 = stages[k].m_0
            m_f = stages[k].m_f

            # f_min for current stage
            if cs_list[k].f_min.enabled and cs_list[k].f_min.value is not None:
                lbf = max(cs_list[k].f_min.value - self.context.delta, 0)
            else:
                lbf = 0

            # gap closing
            lbg += nx*[0]
            ubg += nx*[0]

            if first_loop_node:
                gb_0 = self.x0.get_gb(self.context)
                lbg += gb_0['lbg']
                ubg += gb_0['ubg']

                xb_0 = self.x0.get_xb(self.context)
                ub_0 = self.x0.get_ub(self.context)
                lbx += [m_0] + xb_0['lbx'] + [lbf] + ub_0['lbu']
                ubx += [m_0] + xb_0['ubx'] + [1]   + ub_0['ubu']

            else: # interior points
                # radius constraint
                lbg.append(0)
                ubg.append(ca.inf)

                if stage_interface: # stage interface nodes
                    max_cs = ConstraintSet.choose_max(cs_list[k-1], cs_list[k])
                    lbx += [m_0] + xb_free['lbx'] + [lbf] + ub_free['lbu']
                    ubx += [m_0] + xb_free['ubx'] + [1]   + ub_free['ubu']
                else: # non-stage interface interior nodes ('regular nodes')
                    max_cs = cs_list[k]
                    if last_loop_node:
                        ub_f = self.xf.get_ub(self.context)
                        lbx += [m_f] + xb_free['lbx'] + [lbf] + ub_f['lbu']
                        ubx += [m_0] + xb_free['ubx'] + [1]   + ub_f['ubu']
                    else:
                        lbx += [m_f] + xb_free['lbx'] + [lbf] + ub_free['lbu']
                        ubx += [m_0] + xb_free['ubx'] + [1]   + ub_free['ubu']

                # rate and q constraints here
                if max_cs.max_tau.enabled and max_cs.max_tau.value is not None:
                    lbg.append(-max_cs.max_tau.value)
                    ubg.append(max_cs.max_tau.value)
                if max_cs.max_body_rate_y.enabled and max_cs.max_body_rate_y.value is not None:
                    lbg.append(-max_cs.max_body_rate_y.value)
                    ubg.append(max_cs.max_body_rate_y.value)
                if max_cs.max_body_rate_z.enabled and max_cs.max_body_rate_z.value is not None:
                    lbg.append(-max_cs.max_body_rate_z.value)
                    ubg.append(max_cs.max_body_rate_z.value)
                if max_cs.max_q.enabled and max_cs.max_q.value is not None: # max q
                    lbg.append(0)
                    ubg.append(max_cs.max_q.value)
                
            if not first_loop_node and not last_loop_node:
                # f_min and alpha constraints here
                if cs_list[k].f_min.enabled and cs_list[k].f_min.value is not None:
                    lbg.append(0)
                    ubg.append(1)
                if cs_list[k].max_alpha.enabled and cs_list[k].max_alpha.value is not None: # max AoA
                    lbg.append(np.cos(cs_list[k].max_alpha.value))
                    ubg.append(1)
        
        # final constraint
        gb_f = self.xf.get_gb(self.context)
        lbg += gb_f['lbg']
        ubg += gb_f['ubg']

        xb_f = self.xf.get_xb(self.context)
        lbx += [m_f] + xb_f['lbx']
        ubx += [m_0] + xb_f['ubx']

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