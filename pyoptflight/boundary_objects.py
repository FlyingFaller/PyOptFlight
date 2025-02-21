from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from solver import Solver  # Only used during type checking
from .functions import *
from .setup import *
import numpy as np

def offset_states(state, solver: "Solver"):
    state, _ = rotate_trajectory(state, np.ones((3)), np.array([0, 0, 1]), -solver.config.delta_nu_init)
    return state

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
    
    # def get_he(self, solver: "Solver"):
    #     if self.state[2] is None:
    #         state = self.state.copy()
    #         state[2] = 0
    #     _, _, _, _, _, _, h, e = state_to_kep(np.array(state), solver.body.mu)
    #     return h, e
        
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

class LatLngBoundary(StateBoundary):
    def __init__(self, lat, lng, alt, v_eps, ub_lng, lb_lng, body: "Body", config: "SolverConfig") -> None:
        """
        Allows representation of lat/lng boundaries that have unknown Longitude, are fully defined,
        or have a Longitude determined at time of landing. 

        Parameters:
            lat (float): Latitude in degrees.
            lng (float or None): Longitude in degrees; if landing this is the Longitude of the landing location
            at the start of the problem; if None the landing/ascent location is unknown.
            alt (float): Altitude above the reference radius.
            v_eps (float): Radial velocity.
            lng_upper (float): Upper bound on longitude (degrees) for phi.
            lng_lower (float): Lower bound on longitude (degrees) for phi.
            body (Body): The body object containing parameters such as r_0 and psi.
            config (SolverConfig): Solver configuration; contains parameters like phi_T0, landing flag, etc.
        """
        self.lat = lat
        self.lng = lng
        self.alt = alt
        self.lng_upper = ub_lng
        self.lng_lower = lb_lng

        self.fully_defined = False if self.lng is None else True

        # Compute state components.
        r = body.r_0 + alt
        theta = np.pi / 2 - np.deg2rad(lat)
        # Determine phi: if lng is not provided or if we are in landing mode, leave phi unknown.
        if not self.fully_defined or config.landing:
            phi = None
        else:
            phi = config.pmerid_offset + np.deg2rad(lng)
        vr = v_eps
        omega = 0
        psi = body.psi

        self.state = [r, theta, phi, vr, omega, psi]

        # Set bounds. For phi, if unknown, use the provided longitude bounds.
        self.ubx = self.state.copy()
        self.lbx = self.state.copy()
        if not self.fully_defined or config.landing:
            if config.landing: # Formulate bounds around predicted landing zone
                self.ubx[2] = config.pmerid_offset + np.deg2rad(lb_lng) + config.T_init*body.psi
                self.lbx[2] = config.pmerid_offset + np.deg2rad(ub_lng) + config.T_init*body.psi
            else: # Formulate bounds around launch zone
                self.ubx[2] = config.pmerid_offset + np.deg2rad(lb_lng)
                self.lbx[2] = config.pmerid_offset + np.deg2rad(ub_lng)
        else: # Enforce strict equality
            self.ubx = self.state.copy()
            self.lbx = self.state.copy()

        # Initialize the base class with the computed state and bounds.
        # super().__init__(state, ubx, lbx)

    def get_gb(self, X, solver: "Solver") -> Dict:
        """
        Defines the constraint for landing: if landing mode is active and longitude is specified,
        then the third state variable (phi) is constrained to match:
        
            X[2] - solver.T * solver.body.psi = config.phi_T0 + deg2rad(lng)
        
        Parameters:
            X: The current state vector in the optimization.
            solver (Solver): The solver instance (contains config and body information).
        
        Returns:
            dict: {g, ubg, lbg} for the defined constraint or empty lists if no constraint applies.
        """
        if solver.config.landing and self.fully_defined:
            target_phi = solver.config.pmerid_offset + np.deg2rad(self.lng)
            # The constraint is defined such that g = 0 when satisfied.
            g = [X[3] - solver.T * solver.body.psi - target_phi]
            tol = solver.config.constraints_tol
            ubg = [tol]
            lbg = [-tol]
            return {"g": g, "ubg": ubg, "lbg": lbg}
        else:
            return {"g":[], "ubg": [], "lbg": []}

    def get_x_range(self, config: "SolverConfig", body: "Body", npoints=1000) -> Dict:
        if not self.fully_defined: # Longitude (by extension phi) is truely unknown
            phi_range = np.linspace(self.lbx[2], self.ubx[2], npoints)
            x_list  = np.zeros((npoints, 6))
            for k, phi in enumerate(phi_range):
                x = np.array([self.state[0], self.state[1], phi, self.state[3], self.state[4], self.state[5]])
                x_list[k] = x
            return {"x": x_list, "axis": np.array([0, 0, 1])}
        
        elif config.landing: # phi can be determined from T_init
            # For landing mode, generate a guess using a prescribed formula.
            x = self.state.copy()
            # Update phi according to the landing initialization.
            x[2] = config.T_init * body.psi + config.pmerid_offset + np.deg2rad(self.lng)
            return {"x": np.array([x]), "axis": np.array([0, 0, 1])}
        else:
            return {"x": np.array([self.state]), "axis": np.array([0, 0, 1])}

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
            h_curr, e_curr = sym_state_to_he(X[1:], solver.body.mu)
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
    
    # def get_he(self, solver: "Solver"):
    #     _, h, e = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, 0, solver.body.mu)
    #     return h, e
        
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
        