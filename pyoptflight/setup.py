from .functions import *
from dataclasses import dataclass
from typing import Optional, List, Callable
import casadi as ca

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
      - max_q, max_alpha, max_body_rate_y, max_body_rate_z, max_tau, f_min
    """
    CONSTRAINT_NAMES = ['max_q', 'max_alpha', 'max_body_rate_y', 'max_body_rate_z', 'max_tau', 'f_min']
    def __init__(self,
                 max_q: Optional[float] = None,
                 max_alpha: Optional[float] = None,
                 max_body_rate_y: Optional[float] = None,
                 max_body_rate_z: Optional[float] = None,
                 max_tau: Optional[float] = None,
                 f_min: Optional[float] = None):
        self.max_q         = Constraint(max_q)
        self.max_alpha     = Constraint(max_alpha)
        self.max_body_rate_y = Constraint(max_body_rate_y)
        self.max_body_rate_z  = Constraint(max_body_rate_z)
        self.max_tau     = Constraint(max_tau)
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
            match force_source:
                case 'global':
                    base = global_cs[name]
                case 'stage':
                    base = stage_cs[name]
                case _:
                    # Default behavior: use stage if a value is defined, otherwise use global.
                    base = stage_cs[name] if stage_cs[name].value is not None else global_cs[name]
            # Construct a new Constraint using both the value and the enabled flag.
            setattr(merged, name, Constraint(base.value, base.enabled))
        return merged

    @classmethod
    def choose_max(cls, cs_1: "ConstraintSet", cs_2: "ConstraintSet") -> "ConstraintSet":
        """
        Chooses 'most strict' constraint for each constraint in a set. 
        """
        new_cs = cls()
        for name in cls.CONSTRAINT_NAMES:
            c1 = cs_1[name]
            c2 = cs_2[name]
            if c1.enabled and c2.enabled and c1.value is not None and c2.value is not None:
                new_value = min(c1.value, c2.value)
                new_enable = True
            elif c1.enabled and c1.value is not None:
                new_value = c1.value
                new_enable = True
            elif c2.enabled and c2.value is not None:
                new_value = c2.value
                new_enable = True
            else:
                new_value = None
                new_enable = False
            new_cs.set(name, new_value, new_enable)
        return new_cs

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
            self.Rg = atm_params.get("Rg")
            self.cutoff_altitude = atm_params.get("cutoff_altitude")
            self.color = atm_params.get("color", "gray")
            T_data = atm_params.get("T", 273.15) # Get temperature, default to constant 0 C
            if isinstance(T_data, str):
                data_table = load_csv("defaults/"+T_data)
                alt_range = data_table['data'][:, 0].astype(float) # rows
                temp_range = data_table['data'][:, 1:].astype(float) # table of data 
                # linear seems more robust, bspline may be more accurate when evaluations are garunteed to be in range
                temp_lut = ca.interpolant('coeffs_lut','linear', [alt_range], temp_range)
                self.T = lambda altitude: temp_lut(altitude)
            else:
                self.T = lambda altitude: float(T_data)

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
        self.omega_0 = body_params.get("omega_0")
        atm_params = body_params.get("atm", {})
        self.atm = self.Atmosphere(atm_params) if atm_params else None
        self.meshpath = body_params.get("meshpath")

class Stage(AutoRepr):
    """Stores rocket stage mass, aerodynamics, propulsion, and limits."""
    class Aerodynamics(AutoRepr):
        def __init__(self, aero_params: dict, folder_path: str = None):
            self.A_ref = aero_params.get("A_ref", 1.0)
            self.C_D  = aero_params.get("C_D", 0.0)
            self.C_L  = aero_params.get("C_L", 0.0)

            C_A_data  = aero_params.get("C_A", 0.0)
            C_Ny_data = aero_params.get("C_Ny", 0.0)
            C_Nz_data = aero_params.get("C_Nz", 0.0)
            self.C_A: Callable
            self.C_Ny: Callable
            self.C_Nz: Callable
            for attr, data in zip(['C_A', 'C_Ny', 'C_Nz'], [C_A_data, C_Ny_data, C_Nz_data]):
                if isinstance(data, str):
                    data_table = load_csv(folder_path+"\\"+data)
                    mach_range = data_table['header'][1:].astype(float) # columns
                    angle_range = data_table['data'][:, 0].astype(float) # rows
                    coeffs = data_table['data'][:, 1:].astype(float) # table of data 
                    coeffs_flat = coeffs.ravel(order='F')
                    # linear seems more robust, bspline may be more accurate when evaluations are garunteed to be in range
                    coeffs_lut = ca.interpolant('coeffs_lut','linear',[angle_range, mach_range], coeffs_flat)
                    setattr(self, attr, lambda mach, angle: coeffs_lut([angle, mach]))
                else:
                    setattr(self, attr, lambda mach, angle: float(data))

    class Propulsion(AutoRepr):
        def __init__(self, prop_params):
            # TODO:  Add better modeling of pressure variant engine performance
            F = prop_params.get("F")
            Isp = prop_params.get("Isp")
            self.F_SL    = prop_params.get("F_SL", F)
            self.F_vac   = prop_params.get("F_vac", F)
            self.Isp_SL  = prop_params.get("Isp_SL", Isp)
            self.Isp_vac = prop_params.get("Isp_vac", Isp)
        
    def __init__(self, stage_params):
        if not isinstance(stage_params, dict):
            raise TypeError("Input must be a string or a dictionary")

        self.name = stage_params.get("name", None)
        self.description = stage_params.get("description", None)
        self.folder_path = stage_params.get("folder_path", None)

        self.m_0 = stage_params.get("m_0")
        self.m_f = stage_params.get("m_f")
        self.aero = self.Aerodynamics(stage_params.get("aero", {}))
        self.prop = self.Propulsion(stage_params.get("prop"))

        constraints = stage_params.get("constraints", {})
        self.constraints = ConstraintSet(
            max_q           = constraints.get("max_q"),
            max_alpha       = constraints.get("max_alpha"),
            max_body_rate_y = constraints.get("max_body_rate_y"),
            max_body_rate_z = constraints.get("max_body_rate_z"),
            max_tau         = constraints.get("max_tau"),
            f_min           = constraints.get("f_min")
        )

        self.T_init = stage_params.get("T_init")
        self.T_min = stage_params.get("T_min")
        self.T_max = stage_params.get("T_max")
        self.N = stage_params.get("N")

class Vehicle(AutoRepr):
    def __init__(self, stages: List[Stage], name: str = None, description: str = None):
        self.name = name
        self.description = description
        self.stages = stages

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> Stage:
        return self.stages[index]

    def __iter__(self):
        return iter(self.stages)
    
    @classmethod
    def load_vehicle(cls, name:str) -> "Vehicle":
        vehicle_path = f"defaults\\vehicles\\{name}"
        vehicle_dict = load_json(vehicle_path+"\\vehicle.json")
        vehicle_name = vehicle_dict.get("name", None)
        vehicle_description = vehicle_dict.get("description", None)
        stage_params_list = vehicle_dict.get('stages')
        stages_objects: list[Stage] = [Stage(stage_data, vehicle_path) for stage_data in stage_params_list]
        return cls(name=vehicle_name, description=vehicle_description, stages=stages_objects)

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
            max_q         = kwargs.get('max_q'),
            max_alpha     = kwargs.get('max_alpha'),
            max_body_rate_y = kwargs.get('max_body_rate_y'),
            max_body_rate_z  = kwargs.get('max_body_rate_z'),
            max_tau     = kwargs.get('max_tau'),
            f_min         = kwargs.get('f_min')
        )
        self.force_constraints = kwargs.get('force_constraints')

