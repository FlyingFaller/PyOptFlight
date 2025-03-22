from typing import TYPE_CHECKING, Dict, Callable
if TYPE_CHECKING:
    from solver import Solver  # Only used during type checking
from .functions import *
from .setup import *
import numpy as np
from scipy.optimize import differential_evolution

def _optimize_extreme(func: Callable, bounds: list[tuple[float, float]]) -> tuple[float, float]:
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

class BoundaryObj(AutoRepr):
    def __init__(self, **kwargs) -> None:
        pass
    def get_x0s(self, solver: "Solver", npoints=100) -> Dict:
        return {'pos': [], 'vel': [], 'axis': []}
    def get_ge(self, Xi, Ui, T_sum, solver: "Solver") -> Dict:
        return {"g": [], "e": []}
    def get_gb(self, solver: "Solver") -> Dict:
        return {"lbg": [], "ubg": []}
    def get_xb(self, solver: "Solver") -> Dict:
        return {"ubx": [], "lbx": []}
    def get_ub(self, solver: "Solver") -> Dict:
        return {"ubu": [], "lbu": []}
    
class FreeBound(BoundaryObj):
    def get_xb(self, solver):
        return {"ubx": 6*[ca.inf],
                "lbx": 6*[-ca.inf]}
    def get_ub(self, solver):
        return {"ubu": [1, ca.pi, ca.pi/2],
                "lbu": [0, -ca.pi, -ca.pi/2]}

class LatLngBound(BoundaryObj):
    def __init__(
        self,
        lat: float, # Latitude in radians
        lng: float, # Longitude in radians
        alt: float, # Altitude in km
        vel: float, # Velocity in km/s
        # f: None | float = None, # Throttle setting. None lets solver determine./
        # atti: tuple[float, ...] = (0, 0), # Vehicle orientation relative to position. Tuple specifies psi and theta relative to r vec.
        ERA0: None | float = None, # Earth Rotation Angle at T=0 in radians. None lets solver determine subject to range.
        ERA0_range: tuple[float, ...] = (0, 2*np.pi), # (min, max) ERA0 in radians. 
    ) -> None:
        self.lat = lat
        self.lng = lng
        self.alt = alt
        self.vel = vel
        # self.f = f
        # self.atti = atti
        self.ERA0 = ERA0
        self.ERA0_range = ERA0_range
    
    def get_x0s(self, solver: "Solver", npoints=100) -> Dict: # For use only in initialization scripts
        # Need to determine possible pos, vel, f, atti and axis value
        T_min = sum(solver.T_min)
        T_max = sum(solver.T_max)
        omega_0 = solver.body.omega_0
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
                npoints = 1

        r = solver.body.r_0 + self.alt
        theta = np.pi/2 - np.deg2rad(self.lat)
        speed = -self.vel if solver.config.landing else self.vel
        phi_range = np.linspace(self.ERA_range[0], self.ERA_range[1], npoints) + np.deg2rad(self.lng)

        poses = np.zeros((npoints, 3))
        vels = np.zeros((npoints, 3))
        # ctrls = np.zeros((npoints, 3))
        
        for i, phi in enumerate(phi_range):
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            poses[i] = np.array([x, y, z])

            vx = speed*np.sin(theta)*np.cos(phi) - omega_0*y
            vy = speed*np.sin(theta)*np.sin(phi) + omega_0*x
            vz = speed*np.cos(theta)
            vels[i] = np.array([vx, vy, vz])

            # f = 0.5 # Default throttle setting
            # psi_ctrl = (phi + self.atti[0] + np.pi) % (2*np.pi) - np.pi
            # theta_ctrl = -np.deg2rad(self.lng) + self.atti[1]
            # ctrls[i] = np.array([f, psi_ctrl, theta_ctrl])
        return {'pos': np.atleast_2d(poses), 
                'vel': np.atleast_2d(vels), 
                # 'ctrl': np.atleast_2d(ctrls), 
                'axis': np.array([0, 0, 1])}

    def get_ge(self, Xi, Ui, T_sum, solver: "Solver") -> Dict:
        x, y, z = Xi[1], Xi[2], Xi[3]
        vx, vy, vz = Xi[4], Xi[5], Xi[6]
        vx_rel = vx + solver.body.omega_0*y
        vy_rel = vy - solver.body.omega_0*x
        g = []
        e = []
        if solver.config.landing or self.ERA0 is None:
            g.append((x**2 + y**2) - ((solver.body.r_0 + self.alt)*np.cos(np.deg2rad(self.lat)))**2) # constraint on length of px, py (pz is known)
            g.append((vx_rel**2 + vy_rel**2) - (self.vel*np.cos(np.deg2rad(self.lat)))**2) # constraint on length of vx, vy (vz is known)
            e += [True, True]
        if self.ERA0 is None:
            if solver.config.landing:
                # g.append(ca.arctan2(y*vx_rel - x*vy_rel, -x*vx_rel - y*vy_rel)) # constraint on velocity direction
                # x.y/(|x||y|) = cos(0) = 1
                g.append((-vx_rel*x - vy_rel*y) - self.vel*np.cos(np.deg2rad(self.lat))*ca.sqrt(x**2+y**2))
            else:
                # g.append(ca.arctan2(x*vy_rel - y*vx_rel, x*vx_rel + y*vy_rel)) # constraint on velocity direction
                g.append((vx_rel*x + vy_rel*y) - self.vel*np.cos(np.deg2rad(self.lat))*ca.sqrt(x**2+y**2))
            e.append(True)
        elif solver.config.landing:
            g.append(ca.arctan2(y, x) - (ca.fmod(self.ERA0 + solver.body.omega_0*T_sum + np.deg2rad(self.lng) + ca.pi, 2*ca.pi) - ca.pi)) # constraint on position angle at time of landing
            # alternate formulation:
            # angle = self.ERA0 + solver.body.omega_0*T_sum + np.deg2rad(self.lng)
            # xdes = ca.cos(angle)
            # ydex = ca.sin(angle)
            # g.append(xdes*x + ydes*y - ca.sqrt(x**2 + y**2))
            # g.append(ca.arctan2(y*vx_rel - x*vy_rel, -x*vx_rel - y*vy_rel)) # constraint on velocity direction
            g.append((-vx_rel*x - vy_rel*y) - self.vel*np.cos(np.deg2rad(self.lat))*ca.sqrt(x**2+y**2))
            e += [True, True]

        # constrain on attitude direction
        psi, theta = Ui[1], Ui[2]
        atti_x = ca.cos(theta)*ca.cos(psi)
        atti_y = ca.cos(theta)*ca.sin(psi)
        atti_z = -ca.sin(theta)
        r = ca.sqrt(x**2 + y**2 + z**2)
        g.append(x*atti_x + y*atti_y + z*atti_z - r)
        e.append(True)

        return {"g": g, "e": e}
    
    def get_gb(self, solver: "Solver") -> Dict:
        lbg = []
        ubg = []
        if self.ERA0 is None:
            lbg += 4*[0]
            ubg += 4*[0]
        elif solver.config.landing:
            lbg += 5*[0]
            ubg += 5*[0]
        return {"lbg": lbg, "ubg": ubg}

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
        px_func = lambda phi: r*np.sin(theta)*np.cos(phi[0])
        py_func = lambda phi: r*np.sin(theta)*np.sin(phi[0])
        px_min, px_max = _optimize_extreme(px_func, [(phi_min, phi_max)])
        py_min, py_max = _optimize_extreme(py_func, [(phi_min, phi_max)])
        pz_min, pz_max = r*np.cos(theta), r*np.cos(theta)

        # Determine min max vx, vy, vz
        speed = -self.vel if solver.config.landing else self.vel
        def vx_func(phi):
            x = np.sin(theta)*np.cos(phi[0])
            y = np.sin(theta)*np.sin(phi[0])
            vx = speed*x  - omega_0*y*r
            return vx
        def vy_func(phi):
            x = np.sin(theta)*np.cos(phi[0])
            y = np.sin(theta)*np.sin(phi[0])
            vy = speed*y + omega_0*x*r
            return vy
        vx_min, vx_max = _optimize_extreme(vx_func, [(phi_min, phi_max)])
        vy_min, vy_max = _optimize_extreme(vy_func, [(phi_min, phi_max)])  
        vz_min, vz_max = speed*np.cos(theta), speed*np.cos(theta)

        return {
            "ubx": [px_max, py_max, pz_max, vx_max, vy_max, vz_max],
            "lbx": [px_min, py_min, pz_min, vx_min, vy_min, vz_min]
        }
    
    def get_ub(self, solver: "Solver"):
        return FreeBound.get_ub(self, solver)
    
class StateBound(BoundaryObj):
    def __init__(self, **kwargs) -> None:
        self.state = kwargs.get("state")
        self.ubx = kwargs.get("ubx")
        self.lbx = kwargs.get("lbx")
        self.pos = kwargs.get("pos")
        self.vel = kwargs.get("vel")
        self.ctrl = kwargs.get("ctrl")
        self.atti = kwargs.get("atti")
        self.f = kwargs.get("f")

    def get_x0s(self, solver: "Solver", npoints=100) -> Dict:
        if self.state is not None:
            return {"pos": np.atleast_2d(self.state[:3]), 
                    "vel": np.atleast_2d(self.state[3:6]), 
                    # "ctrl": np.atleast_2d(self.state[6:9]),
                    "axis": np.array([0, 0, 1])} # May be more correct to make zero vector?
        elif self.pos is not None and self.vel is not None:
            return {"pos": np.atleast_2d(self.pos), 
                    "vel": np.atleast_2d(self.vel), 
                    # "ctrl": np.atleast_2d(self.ctrl),
                    "axis": np.array([0, 0, 1])}
        else:
            if self.ubx == self.lbx:
                return {"pos": self.ubx[:3], "vel": self.ubx[3:6]}
            else:
                poses = np.zeros((npoints, 3))
                vels = np.zeros((npoints, 3))
                # ctrls = np.zeros((npoints, 3/))
                for i in range(npoints):
                    state = self.lbx + (self.ubx - self.lbx)*i/(npoints-1)
                    poses[i] = state[:3]
                    vels[i] = state[3:6]
                    # ctrls[i] = state[6:9]
                axis = np.cross(poses[0], poses[-1])
                axis = axis/np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array([0, 0, 1])
                return {"pos": np.atleast_2d(poses), 
                        "vel": np.atleast_2d(vels), 
                        # "ctrl": np.atleast_2d(ctrls),
                        "axis": axis}            

    def get_xb(self, solver: "Solver") -> Dict:
        if self.ubx is not None and self.lbx is not None:
            if len(self.ubx) == 6 and len(self.lbx) == 6:
                return {"ubx": self.ubx, "lbx": self.lbx}    
        elif self.state is not None:
            if len(self.state) == 6:
                return {"ubx": self.state, "lbx": self.state}
        else:
            self.state = np.concatenate((self.pos, self.vel))
            return {"ubx": self.state, "lbx": self.state}

    def get_ub(self, solver):
        if self.ctrl is not None:
            f_min, f_max = self.ctrl[0], self.ctrl[0]
            atti_min, atti_max = self.ctrl[1:], self.ctrl[1:]
        else:
            udict = FreeBound.get_ub(self, solver)
            f_min = self.f if self.f is not None else udict['lbu'][0]
            atti_min = self.atti if self.atti is not None else udict['lbu'][1:]
            f_max = self.f if self.f is not None else udict['ubu'][0]
            atti_max = self.atti if self.atti is not None else udict['ubu'][1:]
        return {'ubu': f_max + atti_max,
                'lbu': f_min + atti_min}
    
class KeplerianBound(BoundaryObj):
    """
    Allows representation of kerplarian boundaries that are either fully defined or have an unknown True Anamolys.

    Parameters:
        body (Body): celestial body parameters.
        i, Ω, ω: Required Keplarian angles.
        e, a: Optional Keplarian elements.
        ha, hp: Optional Keplarian elements if e, a are not provided.
    """
    def __init__(self, body: Body, i=0, Ω=0, ω=0, **kwargs) -> None:
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
        # if None (default), max range is assigned. Behavior is a bit clunky may clean up later. 
        self.ctrl = kwargs.get("ctrl")
        self.atti = kwargs.get("atti")
        self.f = kwargs.get("f")

    def get_x0s(self, solver: "Solver", npoints=100) -> Dict:
        if self.ν is None:
            ν_range = np.linspace(-np.pi, np.pi, npoints)
            poses = np.zeros((npoints, 3))
            vels = np.zeros((npoints, 3))
            # ctrls = np.zeros((npoints, 3))

            for k, ν in enumerate(ν_range):
                pos_vel, h, e = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, ν, solver.body.mu)
                poses[k] = pos_vel[:3]
                vels[k] = pos_vel[3:6]

        else:
            pos_vel, h, e = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, self.ν, solver.body.mu)
            poses = pos_vel[:3]
            vels = pos_vel[3:6]
        return {'pos': np.atleast_2d(poses), 
                'vel': np.atleast_2d(vels), 
                # 'ctrl': np.atleast_2d(ctrls), 
                'axis': h/np.linalg.norm(h)}
    
    def get_ge(self, Xi, T, solver: "Solver") -> Dict:
        if self.ν is not None:
            return {"g": [], "e": []}
        else:
            x, y, z = Xi[1], Xi[2], Xi[3]
            vx, vy, vz = Xi[4], Xi[5], Xi[6]
            pos = ca.vertcat(x, y, z)
            vel = ca.vertcat(vx, vy, vz)
            h_curr = ca.cross(pos, vel)
            e_curr = ca.cross(vel, h_curr)/solver.body.mu - pos/ca.norm_2(pos)
            _, h_T, e_T = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, 0, solver.body.mu)
            g = []
            e = []
            for i in range(0, 3):
                g.append(h_T[i] - h_curr[i])
                g.append(e_T[i] - e_curr[i])
                e += [True, True]
            return {"g": g, "e": e}
        
    def get_gb(self, solver: "Solver") -> Dict:
        """
        Returns constraints and their bounds.
        
        Parameters:
            X: Optimization variable (state vector) at which the constraint is evaluated.
            solver (Solver): The solver instance containing problem configuration.
        Returns:
            dict of lists: {g, ubg, lbg} -- lists of constraints and their corresponding upper and lower bounds.
        """
        if self.ν is not None:
            return {"ubg": [], "lbg": []}
        else:
            tol = solver.config.constraints_tol
            return {"ubg": 6*[tol], "lbg": 6*[-tol]}
        
    def get_xb(self, solver: "Solver") -> Dict:
        """
        Returns:
            dict of lists: {ubx, lbx} -- the upper and lower bounds on the state vector.
        """
        if self.f is not None:
            f_min = self.f
            f_max = self.f
        else:
            f_min = 0
            f_max = 1

        if self.ν is None:
            ν_range = (-np.pi, np.pi)
            ubx = []
            lbx = []
            for i in range(6):
                coord_func = lambda ν: kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, ν[0], solver.body.mu)[0][i]
                min_val, max_val = _optimize_extreme(coord_func, [ν_range])
                lbx.append(min_val)
                ubx.append(max_val)
            return {"ubx": ubx, "lbx": lbx}

        else:
            pos_vel, _, _ = kep_to_state(self.e, self.a, self.i, self.ω, self.Ω, self.ν, solver.body.mu)
            return {"ubx": pos_vel.tolist(), "lbx": pos_vel.tolist()}
        
    def get_ub(self, solver: "Solver"):
        return StateBound.get_ub(self, solver)
        