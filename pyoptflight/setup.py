from .functions import *
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Constraint():
    """
    Holds a numerical constraint value and whether it is active.
    If a value is provided, the constraint automatically becomes enabled unless explicitly set otherwise.
    """
    value: Optional[float] = None
    enabled: Optional[bool] = None

    def __post_init__(self):
        # Automatically enable if a value is provided (unless enabled is explicitly set)
        if self.value is not None:
            if self.enabled is None:
                self.enabled = True
        else:
            if self.enabled is None:
                self.enabled = False

class ConstraintSet(AutoRepr):
    """
    Groups all constraint objects together.
    Constraints include:
      - q_max, alpha_max, gamma_dot_max, beta_dot_max, f_dot_max, f_min
    """
    CONSTRAINT_NAMES = ['q_max', 'alpha_max', 'gamma_dot_max', 'beta_dot_max', 'f_dot_max', 'f_min']
    def __init__(self,
                 q_max: Optional[float] = None,
                 alpha_max: Optional[float] = None,
                 gamma_dot_max: Optional[float] = None,
                 beta_dot_max: Optional[float] = None,
                 f_dot_max: Optional[float] = None,
                 f_min: Optional[float] = None):
        self.q_max         = Constraint(q_max)
        self.alpha_max     = Constraint(alpha_max)
        self.gamma_dot_max = Constraint(gamma_dot_max)
        self.beta_dot_max  = Constraint(beta_dot_max)
        self.f_dot_max     = Constraint(f_dot_max)
        self.f_min         = Constraint(f_min)

    def toggle(self, constraint_name: str, enabled: bool):
        """Enable or disable a specific constraint by its name."""
        if hasattr(self, constraint_name):
            getattr(self, constraint_name).enabled = enabled
        else:
            raise ValueError(f"Constraint '{constraint_name}' does not exist.")

    def set(self, name: str, value: float, enabled: Optional[bool]=None):
        """Directly override a constraint with a new value and state"""
        if hasattr(self, name):
            setattr(self, name, Constraint(value, enabled))
        else:
            raise ValueError(f"Constraint '{name}' does not exist.")    
        
    def set_all_enabled(self, enabled: bool):
        """Enable or disable all constraints in this set."""
        for name in self.CONSTRAINT_NAMES:
            if getattr(self, name).value is not None:
                getattr(self, name).enabled = enabled

    @classmethod
    def merge(cls, global_cs: "ConstraintSet", stage_cs: "ConstraintSet", force_source: Optional[str] = None) -> "ConstraintSet":
        """
        Create a new ConstraintSet where for each constraint, the stage value and enabled flag
        are used if defined; otherwise, the global settings are used.
        """
        merged = cls()  # Create an empty ConstraintSet to populate.
        for name in cls.CONSTRAINT_NAMES:
            if force_source == 'global':
                base = global_cs[name]
            elif force_source == 'stage':
                base = stage_cs[name]
            else:
                # Default behavior: use stage if a value is defined, otherwise use global.
                base = stage_cs[name] if stage_cs[name].value is not None else global_cs[name]
            # Construct a new Constraint using both the value and the enabled flag.
            setattr(merged, name, Constraint(base.value, base.enabled))
        return merged

    # May delete TODO
    def __getitem__(self, constraint_name: str) -> Constraint:
        """Allow dict-like access to constraints."""
        return getattr(self, constraint_name)

class Body(AutoRepr):
    """Stores celestial body parameters."""
    class Atmosphere(AutoRepr):
        def __init__(self, atm_params):
            self.rho_0 = atm_params.get("rho_0")
            self.H = atm_params.get("H")
            self.gamma = atm_params.get("gamma")
            self.R = atm_params.get("R")
            self.C_T = atm_params.get("C_T")
            self.cutoff_altitude = atm_params.get("cutoff_altitude")
            self.color = atm_params.get("color", "gray")

    def __init__(self, body_params):
        default_bodies = load_json(r"defaults/bodies.json")

        if isinstance(body_params, str):
            if body_params in default_bodies:
                body_params = default_bodies[body_params]
            else:
                raise ValueError(f"Unknown default body: {body_params}")
        elif not isinstance(body_params, dict):
            raise TypeError("Input must be a string or a dictionary")

        self.r_0 = body_params.get("r_0")
        self.g_0 = body_params.get("g_0")
        self.mu = body_params.get("mu")
        self.psi = body_params.get("psi")
        atm_params = body_params.get("atm", {})
        self.atm = self.Atmosphere(atm_params) if atm_params else None
        self.meshpath = body_params.get("meshpath")

class Stage(AutoRepr):
    """Stores rocket stage mass, aerodynamics, propulsion, and limits."""
    class Aerodynamics(AutoRepr):
        def __init__(self, aero_params):
            self.C_D = aero_params.get("C_D")
            self.C_L = aero_params.get("C_L")
            self.A_ref = aero_params.get("A_ref")
        
    class Propulsion(AutoRepr):
        def __init__(self, prop_params):
            # TODO:  Add better modeling of pressure variant engine performance
            F = prop_params.get("F")
            Isp = prop_params.get("Isp")
            self.F_SL = prop_params.get("F_SL", F)
            self.F_vac = prop_params.get("F_vac", F)
            self.Isp_SL = prop_params.get("Isp_SL", Isp)
            self.Isp_vac = prop_params.get("Isp_vac", Isp)
        
    def __init__(self, stage_params):
        file_path = r"defaults/stages.json"
        example_stages = load_json(file_path)

        if isinstance(stage_params, str):
            if stage_params in example_stages:
                stage_params = example_stages[stage_params]
            else:
                raise ValueError(f"Unknown default body: {stage_params}")
        elif not isinstance(stage_params, dict):
            raise TypeError("Input must be a string or a dictionary")

        self.m_0 = stage_params.get("m_0")
        self.m_f = stage_params.get("m_f")
        self.aero = self.Aerodynamics(stage_params.get("aero", {}))
        self.prop = self.Propulsion(stage_params.get("prop"))

        constraints = stage_params.get("constraints", {})
        self.constraints = ConstraintSet(
            q_max         = constraints.get("q_max"),
            alpha_max     = constraints.get("alpha_max"),
            gamma_dot_max = constraints.get("gamma_dot_max"),
            beta_dot_max  = constraints.get("beta_dot_max"),
            f_dot_max     = constraints.get("f_dot_max"),
            f_min         = constraints.get("f_min")
        )

        self.T_init = stage_params.get("T_init")
        self.T_min = stage_params.get("T_min")
        self.T_max = stage_params.get("T_max")
        self.N = stage_params.get("N")

    @classmethod
    def load_vehicle(cls, name: str) -> List["Stage"]:
        default_vehicles = load_json(r"defaults/vehicles.json")
        stage_params = default_vehicles[name]
        vehicle = [cls(stage) for stage in stage_params]
        return vehicle

class SolverConfig(AutoRepr):
    def __init__(self, **kwargs):
        ### NLP Settings ###
        self.constraints_tol = kwargs.get('constraints_tol', 1e-6)
        self.solver_tol = kwargs.get('solver_tol', 1e-4)
        self.verbosity = kwargs.get('verbosity', 3)
        self.bound_relax_factor = kwargs.get('bound_relax_factor', 0)
        self.nlp_scaling_method = kwargs.get('nlp_scaling_method', 'none')
        self.mumps_mem_percent = kwargs.get('mumps_mem_percent', 16000)
        self.integration_method = kwargs.get('integration_method', 'cvodes')
        self.max_iter = kwargs.get('max_iter', 500)

        ### Problem Settings ###
        self.landing = kwargs.get('landing', False)
        self.pmerid_offset = kwargs.get('pmerid_offset', 0) # Azimuth angle of prime meridian

        ### Global Defaults ###
        self.T_init = kwargs.get('T_init', 100)
        self.T_min = kwargs.get('T_min', 0)
        self.T_max = kwargs.get('T_max', 600)
        self.N = kwargs.get('N', 300)

        self.global_constraints = ConstraintSet(
            q_max         = kwargs.get('q_max'),
            alpha_max     = kwargs.get('alpha_max'),
            gamma_dot_max = kwargs.get('gamma_dot_max'),
            beta_dot_max  = kwargs.get('beta_dot_max'),
            f_dot_max     = kwargs.get('f_dot_max'),
            f_min         = kwargs.get('f_min')
        )
        self.force_constraints = kwargs.get('force_constraints')

