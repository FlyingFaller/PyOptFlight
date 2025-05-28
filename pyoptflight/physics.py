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

    ### PLANETARY PROPERTIES ###
    def h(self, pos): # px, py, pz
        """Altitude"""
        return ca.norm_2(pos) - self.body.r_0
    
    def g(self, pos): # px, py, pz
        """Gravity vector"""
        g_0 = self.body.g_0
        r_0 = self.body.r_0
        return -g_0*r_0**2*ca.sumsqr(pos)**(-3/2)*pos
    
    def wind(self, pos): # px, py
        """Wind vector in inertial frame"""
        omega_0 = self.body.omega_0
        px, py, pz = ca.vertsplit(pos)
        return ca.vertcat(-omega_0*py, omega_0*px, 0)
    
    def T(self, h): # h(px, py, pz)
        """Returns the temperature at the position"""
        return self.body.atm.T(h)

    def rho(self, h): # h(px, py, pz)
        """Local atmospheric density"""
        return self.body.atm.rho_0*ca.exp(-h/self.body.atm.H)
    
    ### VELOCITY ###
    def v_rel(self, pos, vel): # vx, vy, vz, wind(px, py)
        """Velocity vector relative to atmosphere"""
        wind = self.wind(pos)
        return vel-wind

    ### PROPULSION PERFORMANCE ###
    def F_max(self, h): # h(px, py, pz)
        """Max thrust"""
        F_vac = self.stage.prop.F_vac
        F_SL = self.stage.prop.F_SL
        H = self.body.atm.H
        F_max = F_vac + (F_SL - F_vac)*ca.exp(-h/H)
        return F_max
    
    def f_eff(self, f): # f
        """Effective throttle output"""
        return f - f*ca.fmax(0, ca.fmin(1, (self.f_min - f)/self.delta))
    
    def F_eff(self, h, f): # f_eff(f), F_max(h(px, py, pz))
        """Effective thrust"""
        f_eff = self.f_eff(f)
        F_max = self.F_max(h)
        return f_eff*F_max
    
    def Isp(self, h): # h(px, py, pz)
        """Specific impulse"""
        Isp_vac = self.stage.prop.Isp_vac
        Isp_SL = self.stage.prop.Isp_SL
        H = self.body.atm.H
        Isp = Isp_vac + (Isp_SL - Isp_vac)*ca.exp(-h/H)
        return Isp

    ### VEHICLE ORIENTATION ###    
    def vehicle_basis(self, psi, theta): # psi, theta
        """Returns the vehicle basis vectors."""
        ebx = ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
        eby = ca.vertcat(-ca.sin(psi), ca.cos(psi), 0)
        ebz = ca.vertcat(ca.cos(psi)*ca.sin(theta), ca.sin(psi)*ca.sin(theta), ca.cos(theta))
        return ca.horzcat(ebx, eby, ebz)
 
    def cos_angles(self, v_rel, basis): # vehicle_basis(psi, theta), v_rel(vx, vy, vz, wind(px, py))
        """Cosine angle between air-relative velocity vector and vehicle basis"""
        v_rel = -v_rel if self.config.landing else v_rel
        return (basis.T @ v_rel)/ca.norm_2(v_rel)

    def angles(self, v_rel, basis):
        """Returns the angle (in radians) between air-relative velocity and the vehicle basis"""
        return ca.acos(self.cos_angles(basis, v_rel))

    ### MACH ###
    def mach_sqr(self, h, v_rel): # v_rel(vx, vy, vz, wind(px, py)), T(h(px, py, pz))
        """Returns the square of the mach number"""
        # M = |v|/sqrt(gamma*R*T)
        gamma = self.body.atm.gamma
        Rg = self.body.atm.Rg
        T = self.T(h)
        return ca.sumsqr(v_rel)/(gamma*Rg*T)
    
    def mach(self, h, v_rel):
        """Returns the mach number"""
        return ca.sqrt(self.mach_sqr(v_rel, h))

    ### AERO COEFFICIENTS ###
    def aero_coeffs(self, h, v_rel, basis):
        """Returns C_A, C_Ny, C_Nz aerodynamic coefficients"""
        cos_angles = self.cos_angles(basis, v_rel)
        mach_sqr = self.mach_sqr(v_rel, h)
        C_A = self.stage.aero.C_A(cos_angles[0], mach_sqr)
        C_Ny = self.stage.aero.C_Ny(cos_angles[1], mach_sqr)
        C_Nz = self.stage.aero.C_Nz(cos_angles[2], mach_sqr)
        return ca.vertcat(C_A, C_Ny, C_Nz)

    ### DYNAMIC PRESSURE ###
    def q(self, v_rel, rho):
        """Dynamic pressure q"""
        return 0.5*rho*ca.sumsqr(v_rel)
    
    ### ODE ###
    def ode(self, x, u):
        """ODE state vector x and control vector u"""
        # deconstruct state vectors
        m = x[0]
        pos = x[1:4]
        vel = x[4:7]
        f, psi, theta = u[0], u[1], u[2]

        # Get intermediaries
        A_ref = self.stage.aero.A_ref
        h = self.h(pos)
        F_eff = self.F_eff(h, f)
        Isp = self.Isp(h)
        g = self.g(pos)
        rho = self.rho(h)
        v_rel = self.v_rel(pos, vel)
        basis = self.vehicle_basis(psi, theta)
        
        # Get aerodynamic coefficients
        coeffs = self.aero_coeffs(h, v_rel, basis)
        C_A = coeffs[0]
        C_Ny = coeffs[1]
        C_Nz = coeffs[2]
        
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

        return ca.vertcat(m_dot, vel, v_dot)