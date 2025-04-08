import casadi as ca
from .functions import AutoRepr
from .setup import Stage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .solver import SolverContext

class StagePhysics(AutoRepr):
    def __init__(self, context: "SolverContext", stage: "Stage", f_min: float = 0):
        self.stage = stage
        self.context = context
        self.f_min = f_min

    def h(self, px, py, pz):
        """Altitude"""
        return ca.sqrt(px**2 + py**2 + pz**2) - self.context.body.r_0
    
    def F_max(self, px, py, pz):
        """Max thrust"""
        h = self.h(px, py, pz)
        F_vac = self.stage.prop.F_vac
        F_SL = self.stage.prop.F_SL
        H = self.context.body.atm.H
        F_max = F_vac + (F_SL - F_vac)*ca.exp(-h/H)
        return F_max
    
    def f_eff(self, f):
        """Effective throttle output"""
        return f - f*ca.fmax(0, ca.fmin(1, (self.f_min - f)/self.context.delta))
    
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
        H = self.context.body.atm.H
        Isp = Isp_vac + (Isp_SL - Isp_vac)*ca.exp(-h/H)
        return Isp

    def g(self, px, py, pz):
        """Gravity vector"""
        g_0 = self.context.body.g_0
        r_0 = self.context.body.r_0
        return -g_0*r_0**2*(px**2 + py**2 + pz**2)**(-3/2)*ca.vertcat(px, py, pz)
    
    def rho(self, px, py, pz):
        """Local atmospheric density"""
        h = self.h(px, py, pz)
        return self.context.body.atm.rho_0*ca.exp(-h/self.context.body.atm.H)
    
    def wind(self, px, py, pz):
        """Wind vector in inertial frame"""
        omega_0 = self.context.body.omega_0
        return ca.vertcat(-omega_0*py, omega_0*px, 0)
    
    def v_rel(self, px, py, pz, vx, vy, vz):
        """Velocity vector relative to atmosphere"""
        wind = self.wind(px, py, pz)
        vel = ca.vertcat(vx, vy, vz)
        return vel-wind
    
    def ebx(self, psi, theta):
        """Body frame x basis in intertial frame"""
        return ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
    
    def eby(self, psi, theta):
        """Body frame y basis in intertial frame"""
        return ca.vertcat(-ca.sin(psi), ca.cos(psi), 0)
    
    def ebz(self, psi, theta):
        """Body frame z basis in intertial frame"""
        return ca.vertcat(ca.cos(psi)*ca.sin(theta), ca.sin(psi)*ca.sin(theta), ca.cos(theta))

    def ode(self, x, u):
        """ODE vector of x"""
        # Temp constants, these will be function calls later ig
        C_A = -self.stage.aero.C_D
        C_Ny = self.stage.aero.C_L
        C_Nz = self.stage.aero.C_L
        A_ref = self.stage.aero.A_ref

        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        f, psi, theta = u[0], u[1], u[2]
        F_eff = self.F_eff(px, py, pz, f)
        Isp = self.Isp(px, py, pz)
        g = self.g(px, py, pz)
        rho = self.rho(px, py, pz)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        ebx = self.ebx(psi, theta)

        m_dot = -F_eff/(Isp*9.81e-3)
        # vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[0] + C_Ny*eby[0] + C_Nz*ebz[0])
        # vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[1] + C_Ny*eby[1] + C_Nz*ebz[1])
        # vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*stage.aero.A_ref*ca.sumsqr(v_rel)*(C_A*ebx[2] + C_Ny*eby[2] + C_Nz*ebz[2])
        # Drag only aerodynamics
        vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[0]
        vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[1]
        vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[2]

        return ca.vertcat(m_dot, vx, vy, vz, vx_dot, vy_dot, vz_dot)
    
    def cos_alpha(self, x, u):
        """Cosine of AoA"""
        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        f, psi, theta = u[0], u[1], u[2]
        ebx = self.ebx(psi, theta)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        v_rel = -v_rel if self.context.config.landing else v_rel
        return ca.dot(v_rel, ebx)/ca.norm_2(v_rel)
    
    def q(self, x):
        """Dynamic pressure q"""
        m, px, py, pz, vx, vy, vz = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        rho = self.rho(px, py, pz)
        v_rel = self.v_rel(px, py, pz, vx, vy, vz)
        return 0.5*rho*ca.sumsqr(v_rel)