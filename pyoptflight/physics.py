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
    def h(self, pos):
        """Altitude [float(km)]"""
        return ca.norm_2(pos) - self.body.r_0
    
    def g(self, pos):
        """Gravity vector [vector(m/s2)]"""
        g_0 = self.body.g_0
        r_0 = self.body.r_0
        return -g_0*r_0**2*ca.sumsqr(pos)**(-3/2)*pos
    
    def wind(self, pos):
        """Wind vector in inertial frame [vector(km/s)]"""
        omega_0 = self.body.omega_0
        px, py, pz = ca.vertsplit(pos)
        return ca.vertcat(-omega_0*py, omega_0*px, 0)
    
    def T(self, h):
        """Returns the temperature at the position [float(K)]"""
        return self.body.atm.T(h)

    def rho(self, h): 
        """Local atmospheric density [float(kg/m3)]"""
        return self.body.atm.rho_0*ca.exp(-h/self.body.atm.H)
    
    ### VELOCITY ###
    def v_rel(self, pos, vel): 
        """Velocity vector relative to atmosphere [vector(km/s)]"""
        wind = self.wind(pos)
        return vel-wind

    ### PROPULSION PERFORMANCE ###
    def F_max(self, h):
        """Max thrust [float(MN)]"""
        F_vac = self.stage.prop.F_vac
        F_SL = self.stage.prop.F_ASL
        H = self.body.atm.H
        F_max = F_vac + (F_SL - F_vac)*ca.exp(-h/H)
        return F_max
    
    def f_eff(self, f): 
        """Effective throttle output [float(UNITLESS)]"""
        return f - f*ca.fmax(0, ca.fmin(1, (self.f_min - f)/self.delta))
    
    def F_eff(self, h, f): # f_eff(f), F_max(h(px, py, pz))
        """Effective thrust [float(MN)]"""
        f_eff = self.f_eff(f)
        F_max = self.F_max(h)
        return f_eff*F_max
    
    def Isp(self, h): # h(px, py, pz)
        """Specific impulse [float(s)]"""
        Isp_vac = self.stage.prop.Isp_vac
        Isp_SL = self.stage.prop.Isp_ASL
        H = self.body.atm.H
        Isp = Isp_vac + (Isp_SL - Isp_vac)*ca.exp(-h/H)
        return Isp

    ### VEHICLE ORIENTATION ###    
    def vehicle_basis(self, psi, theta): # psi, theta
        """Returns the vehicle basis vectors [2tensor(UNITLESS)]"""
        ebx = ca.vertcat(ca.cos(psi)*ca.cos(theta), ca.sin(psi)*ca.cos(theta), -ca.sin(theta))
        eby = ca.vertcat(-ca.sin(psi), ca.cos(psi), 0)
        ebz = ca.vertcat(ca.cos(psi)*ca.sin(theta), ca.sin(psi)*ca.sin(theta), ca.cos(theta))
        return ca.horzcat(ebx, eby, ebz)
 
    def cos_angles(self, v_rel, basis): # vehicle_basis(psi, theta), v_rel(vx, vy, vz, wind(px, py))
        """Cosine angle between air-relative velocity vector and vehicle basis [vec(cos(rad))]"""
        v_rel = -v_rel if self.config.landing else v_rel
        return (basis.T @ v_rel)/ca.norm_2(v_rel)

    def angles(self, v_rel, basis):
        """Returns the angle (in radians) between air-relative velocity and the vehicle basis [vec(rad)]"""
        return ca.acos(self.cos_angles(v_rel, basis))

    ### MACH ###
    def mach_sqr(self, h, v_rel): # v_rel(vx, vy, vz, wind(px, py)), T(h(px, py, pz))
        """Returns the square of the mach number [float(UNITLESS)]"""
        # M = |v|/sqrt(gamma*R*T)
        gamma = self.body.atm.gamma # UNITLESS
        Rg = self.body.atm.Rg # J/kg-K
        T = self.T(h) # K
        return ca.sumsqr(1000.0*v_rel)/(gamma*Rg*T) 
    
    def mach(self, h, v_rel):
        """Returns the mach number [float(UNITLESS)]"""
        return ca.sqrt(self.mach_sqr(h, v_rel))

    ### AERO COEFFICIENTS ###
    def axial_normal_coeffs(self, h, v_rel, basis):
        """Returns C_A, C_Ny, C_Nz aerodynamic coefficients [vector(UNITLESS)]"""
        cos_angles = self.cos_angles(v_rel, basis)
        mach_sqr = self.mach_sqr(h, v_rel)
        C_A  = self.stage.aero.C_A(cos_angles[0], mach_sqr)
        C_Ny = self.stage.aero.C_Ny(cos_angles[1], mach_sqr)
        C_Nz = self.stage.aero.C_Nz(cos_angles[2], mach_sqr)
        return ca.vertcat(C_A, C_Ny, C_Nz)

    def lift_drag_coeffs(self, h, v_rel, basis):
        """Returns C_L, C_D, C_S aerodynamic coefficients [vector(UNITLESS)]"""
        cos_angles = self.cos_angles(v_rel, basis)
        mach_sqr = self.mach_sqr(h, v_rel)
        # print(cos_angles[0])
        # print(type(cos_angles[0]))
        # print(mach_sqr)
        # print(type(mach_sqr))
        C_L = self.stage.aero.C_L(cos_angles[0], mach_sqr)
        C_D = self.stage.aero.C_D(cos_angles[0], mach_sqr)
        C_S = self.stage.aero.C_S(cos_angles[0], mach_sqr)
        return ca.vertcat(C_L, C_D, C_S)

    ### AERO FORCES ###
    def axial_normal_force(self, h, pos, vel, basis):
        """Returns the sum of the aerodynamic forces using C_A, C_Ny, C_Nz [vector(MN)]"""
        A_ref = self.stage.aero.A_ref
        rho = self.rho(h)
        v_rel = self.v_rel(pos, vel)
        ebx, eby, ebz = basis[0:3], basis[3:6], basis[6:9]
        coeffs = self.axial_normal_coeffs(h, v_rel, basis)
        C_A, C_Ny, C_Nz = coeffs[0], coeffs[1], coeffs[2]
        return 0.5*rho*A_ref*ca.sumsqr(v_rel)*(C_A*ebx + C_Ny*eby + C_Nz*ebz)

    def lift_drag_force(self, h, pos, vel, basis):
        """Returns the sum of the aerodynamic forces using C_L, C_D, C_S [vector(MN)]"""
        A_ref = self.stage.aero.A_ref
        rho = self.rho(h)
        v_rel = self.v_rel(pos, vel)
        ebx = basis[0:3]
        coeffs = self.lift_drag_coeffs(h, v_rel, basis)
        C_L, C_D, C_S = coeffs[0], coeffs[1], coeffs[2]

        v_dir = v_rel/ca.norm_2(v_rel)
        drag_dir = -v_dir
        lift_dir_unnorm = ebx - v_dir*ca.dot(ebx, v_dir)

        # Effectively an if-else statement to prevent div by zero
        x_input = ca.MX.sym('x_input', 3)
        norm_x = ca.norm_2(x_input)
        f_norm = ca.Function('f_norm', 
                             [x_input], 
                             [ca.if_else(norm_x < 1e-9, ca.MX.zeros(3), x_input/norm_x)])
        
        lift_dir = f_norm(lift_dir_unnorm)
        slip_dir = ca.cross(lift_dir, drag_dir)
        return 0.5*rho*A_ref*ca.sumsqr(v_rel)*(C_D*drag_dir + C_L*lift_dir + C_S*slip_dir)

    ### DYNAMIC PRESSURE ###
    def q(self, v_rel, rho):
        """Dynamic pressure q [float(MPa)]"""
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
        h = self.h(pos)
        F_eff = self.F_eff(h, f)
        Isp = self.Isp(h)
        g = self.g(pos)
        basis = self.vehicle_basis(psi, theta)
        ebx = basis[0:3]

        # Calculate forces
        F_thrust = F_eff*ebx
        match self.config.aero_model:
            case "axial_normal":
                F_aero = self.axial_normal_force(h, pos, vel, basis)
            case "lift_drag":
                F_aero = self.lift_drag_force(h, pos, vel, basis)
            case _:
                raise NotImplementedError(f"self.config.aero_model: {self.config.aero_model} is not an implemented.")

        # Calculate derivatives
        m_dot = -F_eff/(Isp*9.81e-3)
        v_dot = g + (F_thrust + F_aero)/m
        # Drag only aerodynamics
        # vx_dot = g[0] + F_eff/m*ebx[0] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[0]
        # vy_dot = g[1] + F_eff/m*ebx[1] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[1]
        # vz_dot = g[2] + F_eff/m*ebx[2] + 0.5/m*rho*A_ref*ca.norm_2(v_rel)*C_A*v_rel[2]

        return ca.vertcat(m_dot, vel, v_dot)