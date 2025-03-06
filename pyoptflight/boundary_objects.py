from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from solver import Solver  # Only used during type checking
from .functions import *
from .setup import *
import numpy as np
from scipy.optimize import differential_evolution

class BoundaryObj(AutoRepr):
    """
    Boundary Object Template
    """
    def __init__(self, **kwargs) -> None:
        pass
    def get_xb(self, X, solver: "Solver") -> Dict:
        pass
    def get_gb(self, X, solver: "Solver") -> Dict:
        pass
    def get_x_range(self, config: "SolverConfig", body: "Body", npoints=1000) -> Dict:
        pass

# we desire a boundary object that has the following features:
# Lat Lng Boundaries:
# - Must specify a lat and a lng 
# - Can specify a time or leave as None
#   - Leaving time as None results in an unknown position/velocity/attitude
#   - Maybe specify a celestial angular offset instead? Leaving None implise unknown.
#   - In modern astronomy we use ERA (Earth Rotation Angle) to specify the orientation 
#       which can be mapped to times like TAI, UTC, UT1, etc. May be good to use this definition
#   - Probably best to specify a ERA for T=0 in the Body object for consistency between ascent/landing 
#   - Ok what we want is to attach ERA_0 to the LatLng obj. This can be None or defined. Also have a ERA_range 
#       which can be be None or a tuple of (min, max)
# - May specify an ERA_0: float or None
# - May specify an ERA_range: tuple(float, float) or None
# - Must specify an altitude
# - Must specify a velocity
# - Need to get out: state bounds, constraints, and constraint bounds for all states except mass

class LatLngBound(BoundaryObj):
    def __init__(
        self,
        lat: float, # Latitude in radians
        lng: float, # Longitude in radians
        alt: float, # Altitude in km
        vel: float, # Velocity in km/s
        f: None | float = None, # Throttle setting. None lets solver determine.
        atti: None | tuple[float, ...] = (0, 0), # Vehicle orientation relative to position. None lets solver determine subject to range. Tuple specifies exact (psi, theta).
        atti_range: tuple[tuple[float, ...], ...] = ((-np.pi, np.pi), (0, np.pi)), # (min, max) for psi and theta in radians.
        ERA0: None | float = None, # Earth Rotation Angle at T=0 in radians. None lets solver determine subject to range.
        ERA0_range: tuple[float, ...] = (0, 2*np.pi), # (min, max) ERA0 in radians. 
    ) -> None:
        self.lat = lat
        self.lng = lng
        self.alt = alt
        self.vel = vel
        self.f = f
        self.atti = atti
        self.atti_range = atti_range
        self.ERA0 = ERA0
        self.ERA0_range = ERA0_range

    def _optimize_extreme(func, bounds):
        """
        Given a scalar function `func(x)` and bounds (list of (min, max) for each parameter),
        returns the (min, max) value over the domain.
        """
        # Find minimum
        res_min = differential_evolution(func, bounds)
        min_val = res_min.fun
        # Find maximum by minimizing the negative
        res_max = differential_evolution(lambda x: -func(x), bounds)
        max_val = -res_max.fun
        return min_val, max_val
    def get_x0s(): # For use only in initialization scripts
        pass

    def get_g():
        pass
    
    def get_gb():
        pass

    def get_xb(self, solver: "Solver") -> Dict:
        T_min = sum(solver.T_min)
        T_max = sum(solver.T_max)
        omega_0 = solver.body.omega_0

        # Determine ERA_range (may be [ERA0, ERA0])
        if solver.config.landing: # landing
            if self.ERA0 is not None:
                self.ERA_range = (self.ERA0 + omega_0*T_min, self.ERA0 + omega_0*T_max)
            else:
                self.ERA_range = (self.ERA0_range[0] + omega_0*T_min, self.ERA0_range[1] + omega_0*T_max)
        else: # ascent
            if self.ERA0 is None:
                self.ERA_range = self.ERA0_range
            else:
                self.ERA_range = (self.ERA0, self.ERA0)
        
        # Convert ERA range and lng to a global spherical system
        r = solver.body.r_0 + self.alt
        theta = np.pi/2 - np.deg2rad(self.lat)
        phi_min = self.ERA_range[0] + np.deg2rad(self.lng)
        phi_max = self.ERA_range[1] + np.deg2rad(self.lng)

        # Determine min max x, y, z
        px_func = lambda phi: r*np.sin(theta)*np.cos(phi)
        py_func = lambda phi: r*np.sin(theta)*np.sin(phi)
        px_min, px_max = self._optimize_extreme(px_func, [(phi_min, phi_max)])
        py_min, py_max = self._optimize_extreme(py_func, [(phi_min, phi_max)])
        pz_min, pz_max = r*np.cos(theta), r*np.cos(theta)

        # Determine min max vx, vy, vz
        if self.atti is None:
            dpsi_min, dpsi_max = self.atti_range[0]
            dtheta_min, dtheta_max = self.atti_range[1]
        else:
            dpsi_min, dpsi_max = self.atti[0], self.atti[0]
            dtheta_min, dtheta_max = self.atti[1], self.atti[1]
        speed = -self.vel if solver.config.landing else self.vel
        def vx_func(x):
            dpsi, dtheta, phi = x
            x = np.sin(theta + dtheta)*np.cos(phi + dpsi)
            y = np.sin(theta + dtheta)*np.sin(phi + dpsi)
            vx = speed*x  + omega_0*y*r
            return vx
        def vy_func(x):
            dpsi, dtheta, phi = x
            x = np.sin(theta + dtheta)*np.cos(phi + dpsi)
            y = np.sin(theta + dtheta)*np.sin(phi + dpsi)
            vy = speed*y - omega_0*x*r
            return vy
        def vz_func(x):
            dpsi, dtheta, phi = x
            vz = speed*np.cos(theta + dtheta)
            return vz
        vx_min, vx_max = self._optimize_extreme(vx_func, [(dpsi_min, dpsi_max), (dtheta_min, dtheta_max), (phi_min, phi_max)])
        vy_min, vy_max = self._optimize_extreme(vy_func, [(dpsi_min, dpsi_max), (dtheta_min, dtheta_max), (phi_min, phi_max)])  
        vz_min, vz_max = self._optimize_extreme(vz_func, [(dpsi_min, dpsi_max), (dtheta_min, dtheta_max), (phi_min, phi_max)])

        # Determine min max psi, theta
        theta_min = -np.deg2rad(self.lng) + dtheta_min
        theta_max = -np.deg2rad(self.lng) + dtheta_max
        psi_min = (phi_min + dpsi_min) % (2*np.pi) - np.pi
        psi_max = (phi_max + dpsi_max) % (2*np.pi) - np.pi
        f_min, f_max = (0, 1) if self.f is None else (self.f, self.f)

        return {
            "ubx": [px_max, py_max, pz_max, vx_max, vy_max, vz_max, f_max, psi_max, theta_max],
            "lbx": [px_min, py_min, pz_min, vx_min, vy_min, vz_min, f_min, psi_min, theta_min]
        }
class StateBoundary(AutoRepr):
    def __init__(self, state, ubx=None, lbx=None) -> None:
        """
        Allows representation of state boundaries that are either fully defined or have an unknown phi.

        Parameters:
            state (list[float or None]): A 6-element state vector.
            ubx (list[float or None]): Upper bounds on state vector; None indicates strict equality.
            lbx (list[float or None]): Lower bounds on state vector; None indicates strict equality.
        """
        # If no bounds are provided, default to a list of None’s
        if ubx is None:
            ubx = [None] * 6
        if lbx is None:
            lbx = [None] * 6

        self.state = state
        # Replace None in bounds with the corresponding state element.
        self.ubx = [state[i] if ubx[i] is None else ubx[i] for i in range(6)]
        self.lbx = [state[i] if lbx[i] is None else lbx[i] for i in range(6)]

        self.fully_defined = False if self.state[2] is None else True

    def get_xb(self, X, solver: "Solver") -> Dict:
        """
        Returns:
            dict of lists: {ubx, lbx} -- the upper and lower bounds on the state vector.
        """
        return {"ubx": self.ubx, 
                "lbx": self.lbx}

    def get_gb(self, X, solver: "Solver") -> Dict:
        """
        Returns constraints and their bounds.
        
        Parameters:
            X: Optimization variable (state vector) at which the constraint is evaluated.
            solver (Solver): The solver instance containing problem configuration.
        Returns:
            dict: {g, ubg, lbg} -- lists of constraint values and their corresponding upper and lower bounds.
        """
        return {"g":[], 
                "ubg": [], 
                "lbg": []}
    
    def get_x_range(self, config: "SolverConfig", body: "Body", npoints=1000) -> Dict:
        if not self.fully_defined: # If phi is an unknown
            phi_range = np.linspace(self.lbx[2], self.ubx[2], npoints) #np array
            x_list =  np.zeros((npoints, 6))
            for k, phi in enumerate(phi_range):
                x = np.array([self.state[0], self.state[1], phi, self.state[3], self.state[4], self.state[5]])
                x_list[k] = x
            return {"x": x_list, 
                    "axis": np.array([0, 0, 1])}
        else:
            return {"x": np.array([self.state]), 
                    "axis": np.array([0, 0, 1])}

# class LatLngBoundary(StateBoundary):
#     def __init__(self, lat, lng, alt, v_eps, ub_lng, lb_lng, body: "Body", config: "SolverConfig") -> None:
#         """
#         Allows representation of lat/lng boundaries that have unknown Longitude, are fully defined,
#         or have a Longitude determined at time of landing. 

#         Parameters:
#             lat (float): Latitude in degrees.
#             lng (float or None): Longitude in degrees; if landing this is the Longitude of the landing location
#             at the start of the problem; if None the landing/ascent location is unknown.
#             alt (float): Altitude above the reference radius.
#             v_eps (float): Radial velocity.
#             lng_upper (float): Upper bound on longitude (degrees) for phi.
#             lng_lower (float): Lower bound on longitude (degrees) for phi.
#             body (Body): The body object containing parameters such as r_0 and psi.
#             config (SolverConfig): Solver configuration; contains parameters like phi_T0, landing flag, etc.
#         """
#         self.lat = lat
#         self.lng = lng
#         self.alt = alt
#         self.lng_upper = ub_lng
#         self.lng_lower = lb_lng

#         self.fully_defined = False if self.lng is None else True

#         # Compute state components.
#         r = body.r_0 + alt
#         theta = np.pi / 2 - np.deg2rad(lat)
#         # Determine phi: if lng is not provided or if we are in landing mode, leave phi unknown.
#         if not self.fully_defined or config.landing:
#             phi = None
#         else:
#             phi = config.pmerid_offset + np.deg2rad(lng)
#         vr = v_eps
#         omega = 0
#         psi = body.omega_0

#         self.state = [r, theta, phi, vr, omega, psi]

#         # Set bounds. For phi, if unknown, use the provided longitude bounds.
#         self.ubx = self.state.copy()
#         self.lbx = self.state.copy()
#         if not self.fully_defined or config.landing:
#             if config.landing: # Formulate bounds around predicted landing zone
#                 self.ubx[2] = config.pmerid_offset + np.deg2rad(lb_lng) + config.T_init*body.omega_0
#                 self.lbx[2] = config.pmerid_offset + np.deg2rad(ub_lng) + config.T_init*body.omega_0
#             else: # Formulate bounds around launch zone
#                 self.ubx[2] = config.pmerid_offset + np.deg2rad(lb_lng)
#                 self.lbx[2] = config.pmerid_offset + np.deg2rad(ub_lng)
#         else: # Enforce strict equality
#             self.ubx = self.state.copy()
#             self.lbx = self.state.copy()

#     def get_gb(self, X, solver: "Solver") -> Dict:
#         """
#         Defines the constraint for landing: if landing mode is active and longitude is specified,
#         then the third state variable (phi) is constrained to match:
        
#             X[2] - solver.T * solver.body.psi = config.phi_T0 + deg2rad(lng)
        
#         Parameters:
#             X: The current state vector in the optimization.
#             solver (Solver): The solver instance (contains config and body information).
        
#         Returns:
#             dict: {g, ubg, lbg} for the defined constraint or empty lists if no constraint applies.
#         """
#         if solver.config.landing and self.fully_defined:
#             target_phi = solver.config.pmerid_offset + np.deg2rad(self.lng)
#             # The constraint is defined such that g = 0 when satisfied.
#             g = [X[3] - solver.T * solver.body.omega_0 - target_phi]
#             tol = solver.config.constraints_tol
#             ubg = [tol]
#             lbg = [-tol]
#             return {"g": g, "ubg": ubg, "lbg": lbg}
#         else:
#             return {"g":[], "ubg": [], "lbg": []}

#     def get_x_range(self, config: "SolverConfig", body: "Body", npoints=1000) -> Dict:
#         if not self.fully_defined: # Longitude (by extension phi) is truely unknown
#             phi_range = np.linspace(self.lbx[2], self.ubx[2], npoints)
#             x_list  = np.zeros((npoints, 6))
#             for k, phi in enumerate(phi_range):
#                 x = np.array([self.state[0], self.state[1], phi, self.state[3], self.state[4], self.state[5]])
#                 x_list[k] = x
#             return {"x": x_list, "axis": np.array([0, 0, 1])}
        
#         elif config.landing: # phi can be determined from T_init
#             # For landing mode, generate a guess using a prescribed formula.
#             x = self.state.copy()
#             # Update phi according to the landing initialization.
#             x[2] = config.T_init * body.omega_0 + config.pmerid_offset + np.deg2rad(self.lng)
#             return {"x": np.array([x]), "axis": np.array([0, 0, 1])}
#         else:
#             return {"x": np.array([self.state]), "axis": np.array([0, 0, 1])}

class KeplerianBoundary(AutoRepr):
    """
    Allows representation of kerplarian boundaries that are either fully defined or have an unknown True Anamolys.

    Parameters:
        body (Body): celestial body parameters.
        i, Ω, ω: Required Keplarian angles.
        e, a: Optional Keplarian elements.
        ha, hp: Optional Keplarian elements if e, a are not provided.
    """
    def __init__(self, i, Ω, ω, body: Body, **kwargs) -> None:
        ### Should accept e+a or ha+hp
        self.e = kwargs.get("e")
        self.a = kwargs.get("a")
        self.ha = kwargs.get("ha")
        self.hp = kwargs.get("hp")
        self.i = i
        self.Ω = Ω 
        self.ω = ω 
        self.ν = kwargs.get("ν") ### May be None
        if self.e is None or self.a is None:
            ra = self.ha + body.r_0
            rp = self.hp + body.r_0
            self.e = (ra-rp)/(ra+rp)
            self.a = (ra+rp)/2
        elif self.ha is None or self.hp is None:
            self.ha = self.a*(1+self.e) - body.r_0
            self.hp = self.a*(1-self.e) - body.r_0

        self.fully_defined = False if self.ν is None else True

    def get_xb(self, X, solver: "Solver") -> Dict:
        """
        Returns:
            dict of lists: {ubx, lbx} -- the upper and lower bounds on the state vector.
        """
        if not self.fully_defined:
            ν_range = np.linspace(-np.pi, np.pi, 1000)
            x_range = np.zeros((1000, 6))
            for k, ν in enumerate(ν_range):
                x_range[k], _, _ = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, ν, solver.body.mu)
            x_range = change_basis(x_range, None, "cart", "sph")
            x_ulim = np.max(x_range, axis=0)
            x_llim = np.min(x_range, axis=0)
            x_min = [*(x_llim - 0.05*np.abs(x_llim))]
            x_max = [*(x_ulim + 0.05*np.abs(x_ulim))]
            return {"ubx": x_max, "lbx": x_min}
        else:
            x, _, _ = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, self.ν, solver.body.mu)
            x = change_basis(x, None, "cart", "sph")
            return {"ubx": x.tolist(), "lbx": x.tolist()}
        
    def get_gb(self, X, solver: "Solver") -> Dict:
        """
        Returns constraints and their bounds.
        
        Parameters:
            X: Optimization variable (state vector) at which the constraint is evaluated.
            solver (Solver): The solver instance containing problem configuration.
        Returns:
            dict of lists: {g, ubg, lbg} -- lists of constraints and their corresponding upper and lower bounds.
        """
        if self.fully_defined:
            return {"g": [], "ubg": [], "lbg": []}
        else:
            h_curr, e_curr = sym_state_to_he(X[1:7], solver.body.mu)
            _, h_T, e_T = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, 0, solver.body.mu)
            tol = solver.config.constraints_tol
            g = []
            lbg = []
            ubg = []
            for i in range(0, 3):
                g.append(h_T[i] - h_curr[i])
                g.append(e_T[i] - e_curr[i])
                lbg += 2*[-tol]
                ubg += 2*[tol]
            return {"g": g, 
                    "ubg": ubg,
                    "lbg": lbg}

    def get_x_range(self, config: "SolverConfig", body: "Body", npoints=1000) -> Dict:
        if not self.fully_defined:
            ν_range = np.linspace(-np.pi, np.pi, npoints)
            x_list = np.zeros((npoints, 6))
            for k, ν in enumerate(ν_range):
                x_list[k], h, e = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, ν, body.mu)
            x_list = change_basis(x_list, None, "cart", "sph")
            return {"x": x_list, "axis": h/np.linalg.norm(h)}
        else:
            x, h, e = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, self.ν, body.mu)
            x = change_basis(x, None, "cart", "sph")
            return {"x": np.atleast_2d(x), "axis": h/np.linalg.norm(h)}
        