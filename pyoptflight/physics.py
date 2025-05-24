import casadi as ca
from .functions import AutoRepr
from .setup import Stage, Body, SolverConfig, ConstraintSet
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .solver import SolverContext

class StagePhysics(AutoRepr):
    def __init__(self, 
                 stage: "Stage", 
                 body: "Body", 
                 config: "SolverConfig", 
                 constraints: "ConstraintSet", 
                 delta: float):
        self.stage = stage
        self.body = body
        self.config = config
        self.constraints = constraints
        self.delta = delta
        f_min_constr = self.constraints.f_min
        if f_min_constr.enabled and f_min_constr.value is not None:
            f_min = f_min_constr.value
        else:
            f_min = 0
        self.f_min = f_min

    @classmethod
    def create_physics(cls, context: "SolverContext") -> list["StagePhysics"]:
        return [
            cls(context.stages[k], 
                context.body, 
                context.config, 
                context.constraints[k], 
                context.delta)
            for k in range(context.nstages)
        ]

    def h(self, px, py, pz):
        """Altitude"""
        return ca.sqrt(px**2 + py**2 + pz**2) - self.body.r_0
    
    def F_max(self, px, py, pz):
        """Max thrust"""
        h = self.h(px, py, pz)
        F_vac = self.stage.prop.F_vac
        F_SL = self.stage.prop.F_SL
        H = self.body.atm.H
        F_max = F_vac + (F_SL - F_vac)*ca.exp(-h/H)
        return F_max
    
    def f_eff(self, f):
        """Effective throttle output"""
        return f - f*ca.fmax(0, ca.fmin(1, (self.f_min - f)/self.delta))
    
    def F_eff(self, px, py, pz, f):
        """Effective thrust"""
        f_eff = self.f_eff(f)
        F_max = self.F_max(px, py, pz)
        return f_eff*F_max
    
    def Isp(self, px, py, pz):
        """Specific impulse"""
        h = self.h(px, py, pz)
        Isp_vac = self.stage.prop.Isp_vac
        Isp_SL = self.stage.prop.Isp_SL
        H = self.body.atm.H
        Isp = Isp_vac + (Isp_SL - Isp_vac)*ca.exp(-h/H)
        return Isp

    def g(self, px, py, pz):
        """Gravity vector"""
        g_0 = self.body.g_0
        r_0 = self.body.r_0
        return -g_0*r_0**2*(px**2 + py**2 + pz**2)**(-3/2)*ca.vertcat(px, py, pz)
    
    def rho(self, px, py, pz):
        """Local atmospheric density"""
        h = self.h(px, py, pz)
        return self.body.atm.rho_0*ca.exp(-h/self.body.atm.H)
    
    def wind(self, px, py, pz):
        """Wind vector in inertial frame"""
        omega_0 = self.body.omega_0
        return ca.vertcat(-omega_0*py, omega_0*px, 0)
    
    def v_rel(self, px, py, pz, vx, vy, vz):
        """Velocity vector relative to atmosphere"""
        wind = self.wind(px, py, pz)
        vel = ca.vertcat(vx, vy, vz)
        return vel-wind
    
    def vehicle_basis(self, psi, theta):
        """Returns the vehicle basis vectors."""
        ebx = ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
        eby = ca.vertcat(-ca.sin(psi), ca.cos(psi), 0)
        ebz = ca.vertcat(ca.cos(psi)*ca.sin(theta), ca.sin(psi)*ca.sin(theta), ca.cos(theta))
        return ca.horzcat(ebx, eby, ebz)

    def cos_angles(self, px, py, pz, vx, vy, vz, psi, theta):
        """Cosine angle between air-relative velocity vector and vehicle basis"""
        basis = self.vehicle_basis(psi, theta)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        v_rel = -v_rel if self.config.landing else v_rel
        return (basis.T @ v_rel)/ca.norm_2(v_rel)
    
    def T(self, px, py, pz):
        """Returns the temperature at the position"""
        h = self.h(px, py, pz)
        return 273.15 # TODO!!

    def mach_sqr(self, px, py, pz, vx, vy, vz):
        """Returns the square of the mach number"""
        # M = |v|/sqrt(gamma*R*T)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        T = self.T(px, py, pz)
        gamma = self.body.atm.gamma
        Rg = self.body.atm.Rg
        return ca.sumsqr(v_rel)/(gamma*Rg*T)
    
    def mach(self, px, py, pz, vx, vy, vz):
        """Returns the mach number"""
        return ca.sqrt(self.mach_sqr(px, py, pz, vx, vy, vz))

    def aero_coeffs(self, px, py, pz, vx, vy, vz, psi, theta):
        cos_angles = self.cos_angles(px, py, pz, vx, vy, vz, psi, theta)
        mach_sqr = self.mach_sqr(px, py, pz, vx, vy, vz)

    def ode(self, x, u):
        """ODE state vector x and control vector u"""
        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        f, psi, theta = u[0], u[1], u[2]

        coeffs = self.aero_coeffs()
        C_A = coeffs[0]
        C_Ny = coeffs[1]
        C_Nz = coeffs[2]

        A_ref = self.stage.aero.A_ref
        F_eff = self.F_eff(px, py, pz, f)
        Isp = self.Isp(px, py, pz)
        g = self.g(px, py, pz)
        rho = self.rho(px, py, pz)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)

        basis = self.vehicle_basis(psi, theta)
        ebx = basis[0:3]
        eby = basis[3:6]
        ebz = basis[6:9]

        m_dot = -F_eff/(Isp*9.81e-3)

        F_thrust = F_eff/m*ebx
        F_aero = 0.5/m*rho*A_ref*ca.sumsqr(v_rel)*(C_A*ebx + C_Ny*eby + C_Nz*ebz)

        v_dot = g + F_thrust + F_aero
        # Drag only aerodynamics
        # vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[0]
        # vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[1]
        # vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[2]

        return ca.vertcat(m_dot, vx, vy, vz, v_dot)
    
    def alpha(self, px, py, pz, vx, vy, vz, psi, theta):
        """AoA in rads"""
        return ca.acos(self.cos_angles(px, py, pz, vx, vy, vz, psi, theta)[0])

    def q(self, px, py, pz, vx, vy, vz):
        """Dynamic pressure q"""
        rho = self.rho(px, py, pz)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        return 0.5*rho*ca.sumsqr(v_rel)