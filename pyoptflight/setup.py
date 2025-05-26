from .functions import *
from dataclasses import dataclass
from typing import Optional, List, Callable
import casadi as ca
from pathlib import Path

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

class PathManager(AutoRepr):
    """Simple class to keep track of file paths."""
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
    def get_path(self, relative_path: str) -> Path:
        return self.base_path / relative_path
    def load_json_relative(self, relative_path: str) -> dict[str, Any]:
        return load_json(self.get_path(relative_path))
    def load_csv_relative(self, relative_path: str) -> dict[str, np.ndarray]:
        return load_csv(self.get_path(relative_path))

class CallableProp(AutoRepr):
    """This is a callable object that creates interpolants given data."""
    """Maybe this should just be a function that returns a callable?"""
    """I am not convinced..."""
    def __init__(self, 
                 data: int|float|str|dict,
                 path: PathManager,
                 dimension: int, 
                 inputs: list[str] = None,
                 output: str = None, 
                 interp_method: str = 'linear'):
        self.data = data
        self.dimension = dimension
        self.interp_method = interp_method
        self.inputs = inputs
        self.output = output

        if isinstance(data, (int, float)):
            self._callable = lambda *args: float(data)
        elif isinstance(data, dict):
            self._callable = self._create_func_from_dict(data)
        elif isinstance(data, str):
            ext = Path(data).suffix
            if ext == ".json":
                self._callable = self._create_func_from_json(filename=data, path=path)
            elif ext == ".csv":
                self._callable = self._create_func_from_csv(filename=data, path=path)
            else:
                raise NotImplementedError(f"File type of {ext} is not valid.")
        else:
            raise TypeError(f"Unsupported type of {type(data)} for {name} property.")
        
    def _create_func_from_dict(self, data: dict):
        X = np.array(data[self.inputs[0]]) # rows
        if self.dimension == 1:
            fX = np.array(data[self.output]) # data
            lut = ca.interpolant('lut', self.interp_method, [X], fX)
            return lambda x: lut(x)
        elif self.dimension == 2:
            Y = np.array(data[self.inputs[1]])
            fXY = np.array(data[self.output])
            fXY_flat = fXY.ravel(order="F")
            lut = ca.interpolant('lut', self.interp_method, [X, Y], fXY_flat)
            return lambda x: lut(x)
        else:
            raise Exception(f"Unsupported dimension {self.dimension}.")

    def _create_func_from_csv(self, filename: str, path: PathManager):
        data_table = path.load_csv_relative(filename)
        X = data_table['data'][:, 0].astype(float) # rows
        if self.dimension == 1:
            fX = data_table['data'][:, 1].astype(float) # data
            lut = ca.interpolant('lut', self.interp_method, [X], fX)
            return lambda x: lut(x)
        elif self.dimension == 2:
            Y = data_table['header'][1:].astype(float) # columns
            fXY: np.ndarray = data_table['data'][:, 1:].astype(float) # data
            fXY_flat = fXY.ravel(order="F")
            lut = ca.interpolant('lut', self.interp_method, [X, Y], fXY_flat)
            return lambda x, y: lut(x, y)
        else:
            raise Exception(f"Unsupported dimension {self.dimension}.")

    def _create_func_from_json(self, filename: str, path: PathManager):
        data = path.load_json_relative(filename)
        return self._create_func_from_dict(data)

    def __call__(self, *args: Any) -> float:
        return self._callable(*args)
    
class Body(AutoRepr):
    """Stores celestial body parameters."""
    class Atmosphere(AutoRepr):
        def __init__(self, 
                     atm_params: dict, 
                     path: PathManager):
            self.rho_0 = atm_params.get("rho_0")
            self.H = atm_params.get("H")
            self.gamma = atm_params.get("gamma")
            self.Rg = atm_params.get("Rg")
            self.cutoff_altitude = atm_params.get("cutoff_altitude")
            self.color = atm_params.get("color", "gray")
            T_params = atm_params.get("T", 273.15) # Get temperature, default to constant 0 C
            self.T = CallableProp(data=T_params, 
                                  path=path, 
                                  dimension=1, 
                                  inputs=["altitude"], 
                                  output="temperature",
                                  interp_method="linear")

    def __init__(self, 
                 body_params: dict,
                 path: PathManager):
        
        data = body_params.get("data", {})

        self.name = body_params.get("name")
        self.description = body_params.get("description")
        self.type = body_params.get("type")
        self.meshpath = body_params.get("meshpath")

        self.r_0 = data.get("r_0")
        self.g_0 = data.get("g_0")
        self.mu = data.get("mu")
        self.omega_0 = data.get("omega_0", 0)
        atm_params = data.get("atm", {})
        self.atm = self.Atmosphere(atm_params, path) if atm_params else None

    @classmethod
    def load(cls, body_name: str, base_dir: str = "bodies") -> "Body":
        path = PathManager(Path(base_dir)/body_name)
        body_params = path.load_json_relative("body.json")
        return cls(body_params, path)

class Stage(AutoRepr):
    """Stores rocket stage mass, aerodynamics, propulsion, and limits."""
    class Aerodynamics(AutoRepr):
        def __init__(self, 
                     aero_params: dict, 
                     path: PathManager):
            self.A_ref = aero_params.get("A_ref", 1.0)
            self.C_D  = aero_params.get("C_D", 0.0)
            self.C_L  = aero_params.get("C_L", 0.0)

            C_A_data  = aero_params.get("C_A", 0.0)
            C_Ny_data = aero_params.get("C_Ny", 0.0)
            C_Nz_data = aero_params.get("C_Nz", 0.0)
            self.C_A = CallableProp(C_A_data, path, 2, ["angle", "mach"], "C_A", "bspline")
            self.C_Ny = CallableProp(C_Ny_data, path, 2, ["angle", "mach"], "C_Ny", "bspline")
            self.C_Nz = CallableProp(C_Nz_data, path, 2, ["angle", "mach"], "C_Nz", "bspline")

    class Propulsion(AutoRepr):
        def __init__(self, prop_params):
            # TODO:  Add better modeling of pressure variant engine performance
            F = prop_params.get("F")
            Isp = prop_params.get("Isp")
            self.F_SL    = prop_params.get("F_SL", F)
            self.F_vac   = prop_params.get("F_vac", F)
            self.Isp_SL  = prop_params.get("Isp_SL", Isp)
            self.Isp_vac = prop_params.get("Isp_vac", Isp)
        
    def __init__(self, 
                 stage_params: dict, 
                 path: PathManager):
        if not isinstance(stage_params, dict):
            raise TypeError("Input must be a dictionary")

        self.name = stage_params.get("name", None)
        self.description = stage_params.get("description", None)

        self.m_0 = stage_params.get("m_0")
        self.m_f = stage_params.get("m_f")
        self.aero = self.Aerodynamics(stage_params.get("aero", {}), path)
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
    def __init__(self, 
                 stages: List[Stage], 
                 name: str = None, 
                 description: str = None):
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
    def load(cls, vehicle_name: str, base_dir: str = "vehicles") -> "Vehicle":
        path = PathManager(Path(base_dir)/vehicle_name)
        vehicle_params = path.load_json_relative("vehicle.json")
        name = vehicle_params.get("name", None)
        description = vehicle_params.get("description", None)
        stage_params_list = vehicle_params.get('stages')
        stages_objects: list[Stage] = [Stage(stage_params, path) for stage_params in stage_params_list]
        return cls(name=name, description=description, stages=stages_objects)

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

