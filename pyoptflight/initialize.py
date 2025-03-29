from .functions import *
from .setup import *
from .boundary_objects import *
from .solver import Solution, Solver
from typing import List, Callable
import time

def _fix_states(solver: Solver, npoints=250, sep_angle=np.pi/2):
    npoints = 250 if npoints is None else npoints
    sep_angle = np.pi/2 if sep_angle is None else sep_angle
    x0_data = solver.x0.get_x0s(solver, npoints)
    xf_data = solver.xf.get_x0s(solver, npoints)
    x0_points = x0_data['pos']
    xf_points = xf_data['pos']
    x0_axis = x0_data['axis']
    xf_axis = xf_data['axis']
    # print(xf_axis)
    # goal is to find two points that are close to in plane of xf and have a separation close to 90 deg
    nx0 = len(x0_points)
    nxf = len(xf_points)
    cost_table = np.ones((nx0, nxf))
    for i, x0 in enumerate(x0_points):
        for j, xf in enumerate(xf_points):
            xij_axis = np.cross(x0, xf)
            if np.allclose(xij_axis, np.zeros((3))):
                xij_axis = np.zeros((3))
            else:
                xij_axis /= np.linalg.norm(xij_axis)
            if solver.config.landing:
                h_angle = np.arccos(max(min(np.dot(xij_axis, x0_axis), 1), -1))
            else:
                h_angle = np.arccos(max(min(np.dot(xij_axis, xf_axis), 1), -1)) # angle between target plane and plane
            x_angle = np.arctan2(np.dot(np.cross(x0, xf), xij_axis), np.dot(x0, xf)) # signed angle between test points
            
            cost = abs(x_angle - sep_angle) + h_angle
            cost_table[i, j] = cost

    min_ij = np.unravel_index(np.argmin(cost_table, axis=None), cost_table.shape)
    print(f'The minimum cost is {cost_table[min_ij]} rad')
    ### TODO: Implement some sort of preference when there are infinite solutions. Periapsis?
    return {'x0': {'pos': x0_data['pos'][min_ij[0]],
                   'vel': x0_data['vel'][min_ij[0]],
                   'axis': x0_data['axis']},
            'xf': {'pos': xf_data['pos'][min_ij[1]],
                   'vel': xf_data['vel'][min_ij[1]],
                   'axis': xf_data['axis']}}

def _linear_methods(solver: Solver, get_pos: Callable, opts: dict = {}):
    ### WHERE get_pos(t, p0, p1, v0, v1)

    x_data = _fix_states(solver, npoints=opts.get('npoints'), sep_angle=opts.get('sep_angle'))

    ### PREP ###
    spacing = opts.get('spacing', 'T_init')
    if spacing == 'equal':
        seg_props = np.ones((solver.nstages))/solver.nstages
    elif spacing == 'dV':
        dV_stages = np.array([0.5*9.81e-3*(stage.prop.Isp_vac + stage.prop.Isp_SL)*np.log(stage.m_0/stage.m_f) for stage in solver.stages])
        dV_available = np.sum(dV_stages)
        seg_props = dV_stages/dV_available
    elif spacing == 'T_init':
        seg_props = np.array(solver.T_init)/solver.T
    else:
        raise Exception(f"Spacing {spacing} does not exist. Choose from 'equal', 'dV', or T_init (default).")
    
    p0, p1 = x_data['x0']['pos'], x_data['xf']['pos']
    v0, v1 = x_data['x0']['vel'], x_data['xf']['vel']
    v0_mag, v1_mag = np.linalg.norm(v0), np.linalg.norm(v1)

    ts = np.linspace(0, 1, 1000)
    pos_sample = np.array([get_pos(t, p0, p1, v0, v1) for t in ts])

    # Compute the distances between successive positions and the cumulative arc length.
    dists = np.sqrt(np.sum(np.diff(pos_sample, axis=0)**2, axis=1))
    cum_length = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_length[-1]

    # Determine the desired arc-length boundaries for each segment.
    seg_lengths = total_length * seg_props  # each segment’s arc length
    boundaries = np.concatenate(([0], np.cumsum(seg_lengths)))
    seg_times = np.array(solver.T) * seg_props   # allocated time for each segment

    sols = []  # list to hold the solution for each stage/segment
    for k, stage in enumerate(solver.stages):
        seg_start = boundaries[k]
        seg_end   = boundaries[k+1]
        # Equally spaced arc-length positions for this segment.
        seg_arc_positions = np.linspace(seg_start, seg_end, solver.N[k] + 1)
        # Invert the mapping: find the corresponding t values from the cumulative arc-length.
        seg_t = np.interp(seg_arc_positions, cum_length, ts)
        
        # Evaluate the spline at these t values to get segment positions.
        seg_positions = np.array([get_pos(t, p0, p1, v0, v1) for t in seg_t])
        
        # Linearly interpolate the velocity magnitudes for each point.
        seg_vel_mags = (1 - seg_t) * v0_mag + seg_t * v1_mag
        
        seg_f = (1 - seg_t) * 1.0 + seg_t * 0.0

        seg_velocities = []
        seg_controls = []
        # Compute the velocity (and control) at each point from the finite-difference tangent.
        for j in range(solver.N[k] + 1):
            if j < solver.N[k]:
                diff = seg_positions[j + 1] - seg_positions[j]
            else:
                diff = seg_positions[j] - seg_positions[j - 1]
            # Normalize the finite difference to get the direction.
            dir = diff / np.linalg.norm(diff)
            seg_velocities.append(seg_vel_mags[j] * dir)
            
            # Compute a control vector (example: scaled version of the negative/positive direction).
            if solver.config.landing:
                ctrl_dir = -dir
            else:
                ctrl_dir = dir
            psi = np.arctan2(ctrl_dir[1], ctrl_dir[0])
            theta = np.arccos(ctrl_dir[2]) - np.pi/2
            seg_controls.append(np.array([seg_f[j], psi, theta]))
        
        # Compute the mass at each point by linear interpolation between the stage's initial and final masses.
        seg_masses = []
        for t_lin in np.linspace(0, 1, solver.N[k] + 1):
            seg_masses.append((1 - t_lin) * stage.m_0 + t_lin * stage.m_f)

        seg_time = np.linspace(0, seg_times[k], solver.N[k]+1)

        seg_velocities  = np.array(seg_velocities)
        seg_controls = np.array(seg_controls)
        seg_masses = np.array(seg_masses)

        sol = Solution(X=np.block([np.vstack(seg_masses), seg_positions, seg_velocities]).tolist(),
                       U=seg_controls.tolist(),
                       stage=k+1,
                       t=seg_time.tolist(), 
                       )
        sols.append(sol)
    return sols

def linear_interpolation(solver: Solver, opts: dict = {}):
    def get_pos(t, p0, p1, v0, v1):
        r0, r1 = np.linalg.norm(p0), np.linalg.norm(p1)
        r = (1-t)*r0 + t*r1
        d = (1-t)*p0/r0 + t*p1/r1
        d /= np.linalg.norm(d)
        return r*d
    
    sols = _linear_methods(solver, get_pos, opts)
    return {'sols': sols, 'status': None, 'success': True, 'iter_count': 0}

def cubic_bezier_spline(solver: Solver, opts: dict = {}):
    def get_pos(t, p0, p3, v0, v1):
        v0_mag, v1_mag = np.linalg.norm(v0), np.linalg.norm(v1)
        u0 = v0 / v0_mag
        u1 = v1 / v1_mag

        # Use the distance between endpoints as a scale factor
        d = np.linalg.norm(p3 - p0)
        alpha = beta = d / 3.0

        # Define the control points for the Bézier curve
        p1 = p0 + alpha * u0
        p2 = p3 - beta * u1
        pos  = (1 - t)**3 * p0 + 3*(1 - t)**2 * t*p1 + 3*(1 - t)*t**2 * p2 + t**3 * p3

        r0, r1 = np.linalg.norm(p0), np.linalg.norm(p3)
        r = (1-t)*r0 + t*r1

        return r*pos/np.linalg.norm(pos)
    
    sols = _linear_methods(solver, get_pos, opts)
    return {'sols': sols, 'status': None, 'success': True, 'iter_count': 0}

def gravity_turn(solver: Solver, opts: dict = {}):
    def rotate(vec, axis, angle):
        return vec*np.cos(angle) + np.cross(axis, vec)*np.sin(angle) + axis*np.dot(axis, vec)*(1 - np.cos(angle))
    if solver.config.landing:
        raise NotImplementedError("Landing not possible with this method.")

    #############
    ### SETUP ###
    #############

    ### MULTISTAGE SETUP
    ### DETERMINE FIXED STATES GUESS ###
    x_data = _fix_states(solver, npoints=opts.get('npoints'), sep_angle=opts.get('sep_angle')) # Init
    x0_pos, x0_vel, x0_axis = x_data['x0']['pos'], x_data['x0']['vel'], x_data['x0']['axis']
    xf_pos, xf_vel, xf_axis = x_data['xf']['pos'], x_data['xf']['vel'], x_data['xf']['axis']
    rot_angle = np.arccos(np.dot(xf_axis, x0_pos)/np.linalg.norm(x0_pos)) # Angle between xf plane normal an x0
    Yp = np.cross(xf_axis, x0_pos) # Axis normal to x0 and xf
    Yp = Yp/np.linalg.norm(Yp)
    x0_pos_rot = rotate(vec=x0_pos, axis=Yp, angle=np.pi/2-rot_angle)
    x0_vel_rot = rotate(vec=x0_vel, axis=Yp, angle=np.pi/2-rot_angle)
    Xp = x0_pos_rot/np.linalg.norm(x0_pos_rot)
    Zp = xf_axis
    R = np.array([Xp, Yp, Zp])
    x0p_pos = R@x0_pos_rot
    x0p_vel = R@x0_vel_rot
    xfp_pos = R@xf_pos
    xfp_vel = R@xf_vel

    ### COMPUTE INTIALS FOR GTRUN SOLVER ###
    v_init = np.linalg.norm(x0p_vel)
    beta_init = 0.05*ca.pi
    h_init = np.linalg.norm(x0p_pos) - solver.body.r_0
    phi_init = np.arctan2(x0p_pos[1], x0p_pos[0])

    v_T = np.linalg.norm(xfp_vel)
    beta_T = np.arccos(np.dot(xfp_pos, xfp_vel)/(np.linalg.norm(xfp_pos)*np.linalg.norm(xfp_vel)))
    h_T = np.linalg.norm(xfp_pos) - solver.body.r_0
    phi_T = np.arctan2(xfp_pos[1], xfp_pos[0])
    
    #######################
    ### SETUP SYMBOLICS ###
    #######################
    x = ca.SX.sym('[m, v, beta, h, phi]', 5, 1)
    m, v, beta, h, phi = x[0], x[1], x[2], x[3], x[4]
    u = ca.SX.sym('u')
    T = ca.SX.sym('T')

    nx = x.size1() # Number of states (5)
    nu = u.size1() # Number of control vars (1)
    npar = solver.nstages # One time variable per stage

    # Decision variables V has arangment [T0...Tk, U00...U0N,...,Uk0...UkN, X00...X0N+1,...,Xk0...XkN+1]
    V = ca.MX.sym('X', npar + solver.Nu*nu + solver.Nx*nx)
    # Each stages X seperated out shape(nstages, (N+1), nx)
    X = [[V[npar + solver.Nu*nu + nx*(sum(solver.N[0:k]) + k) + i*nx : npar + solver.Nu*nu + nx*(sum(solver.N[0:k]) + k) + (i+1)*nx] 
         for i in range(0, solver.N[k]+1)] 
         for k in range(0, solver.nstages)]
    # All stages U lumped together: shape(nstages, N, nu)
    U = [[V[npar + nu*sum(solver.N[0:k]) + i*nu : npar + nu*sum(solver.N[0:k]) + (i+1)*nu] 
         for i in range(0, solver.N[k])]
         for k in range(0, solver.nstages)]
    
    ##################
    ### INITIALIZE ###
    ##################

    G = [] # Constraint vector G added to in loops
    ubg = []
    lbg = []

    u_min, u_max, u_init = [0.0], [1.0], [0.5]

    x0_min = [solver.stages[0].m_0, v_init, 0.0, h_init, phi_init]
    x0_max = [solver.stages[0].m_0, v_init, 0.5 * ca.pi, h_init, phi_init]
    x0_init = [solver.stages[0].m_0, v_init, beta_init, h_init, phi_init]
    
    xf_min = [solver.stages[-1].m_f, v_T, beta_T, h_T, 0.0]
    xf_max = [solver.stages[-1].m_0, v_T, beta_T, h_T, ca.inf]
    xf_init = [solver.stages[-1].m_f, v_T, beta_T, h_T, phi_T]

    x0 = solver.T_init + solver.Nu*u_init
    lbx = solver.T_min + solver.Nu*u_min
    ubx = solver.T_max + solver.Nu*u_max

    # Linear interpolation between x0_init and xf_init for all stages, duplicates excluded
    x_init = [
        i/(solver.Nx-1)*np.array(xf_init) + (1 - i/(solver.Nx-1))*np.array(x0_init)
        for i in range(0, solver.Nx)
        ]

    ########################
    ### BEGIN STAGE LOOP ###
    ########################
    for k, stage in enumerate(solver.stages):
        # EOMs
        F = stage.prop.F_SL * u
        F_drag = 0.5 * stage.aero.A_ref * stage.aero.C_D * solver.body.atm.rho_0 * ca.exp(-h / solver.body.atm.H) * v ** 2
        r = h + solver.body.r_0
        g = solver.body.g_0 * (solver.body.r_0 / r) ** 2
        v_phi = v * ca.sin(beta)
        vr = v * ca.cos(beta)
        Isp = stage.prop.Isp_vac + (stage.prop.Isp_SL - stage.prop.Isp_vac) * ca.exp(-h / solver.body.atm.H)

        # Build symbolic expressions for ODE right hand side
        m_dot = -(F / (Isp * 9.81e-3))
        v_dot = (F - F_drag) / m - g * ca.cos(beta)
        h_dot = vr
        phi_dot = v_phi / r
        beta_dot = g * ca.sin(beta) / v - phi_dot

        # Create integrator that works from 0 to parameterized dt #
        ode = ca.vertcat(m_dot, v_dot, beta_dot, h_dot, phi_dot)
        dae = {'x': x, 'p': ca.vertcat(u, T), 'ode': T/solver.N[k] * ode}
        int_ops = {'nonlinear_solver_iteration': 'functional'}
        I = ca.integrator('I', 'cvodes', dae, 0.0, 1.0, int_ops)

        # Integrate from 0 to N for the given stage k and enforce EOMs
        for i in range(0, solver.N[k]):
            next_x = I(x0=X[k][i], p=ca.vertcat(U[k][i], V[k]))
            G.append(next_x['xf'] - X[k][i+1]) # we will constrain this part of G to be == 0
            lbg += nx*[0]
            ubg += nx*[0]

        if k+1 < solver.nstages: # If this is not the last stage
            G.append(X[k+1][0][1:] - X[k][-1][1:]) # Force equality between every part of state except mass during staging
            lbg += (nx-1)*[0]
            ubg += (nx-1)*[0]

        # stage specific min/maxes
        x_min = [stage.m_f, v_init, 0.0, 0.0, 0.0]
        x_max = [stage.m_0, ca.inf, ca.pi, ca.inf, ca.inf]


        # linearly interpolate between stage specific m_0 and m_f
        # combine with extracted state from x_init linear interpolation 
        for i in range(0, solver.N[k]+1): # Construct x0, lbx, and ubx
            m_init = i/solver.N[k]*stage.m_f + (1 - i/solver.N[k])*stage.m_0
            x0 += [m_init, *x_init[i+k+sum(solver.N[0:k])][1:].tolist()]

        if k+1 == 1: # if first stage
            lbx += x0_min
            ubx += x0_max
        else:
            lbx += [stage.m_0] + x_min[1:]
            ubx += [stage.m_0] + x_max[1:]

        lbx += (solver.N[k]-1)*x_min
        ubx += (solver.N[k]-1)*x_max

        if k+1 == solver.nstages: # if last stage
            lbx += xf_min
            ubx += xf_max
        else:
            lbx += [stage.m_f] + x_min[1:]
            ubx += [stage.m_f] + x_max[1:]

    ### Create optimization func ###
    # Maximize remaining mass of last stage normalized by mass of last stage
    opt_func = (solver.stages[-1].m_0 - X[-1][-1][0])/(solver.stages[-1].m_0 - solver.stages[-1].m_f)

    #############
    ### SOLVE ###
    #############

    nlp = {'x': V, 'f': opt_func, 'g': ca.vertcat(*G)}
    nlp_opts = {'print_time': True, 'ipopt': 
                {'tol': solver.config.solver_tol, 'print_level': 5, 
                'max_iter': solver.config.max_iter, 'sb': 'yes'}}
    nlpsolver = ca.nlpsol('nlpsolver', 'ipopt', nlp, nlp_opts)

    result = nlpsolver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    #######################
    ### EXTRACT RESULTS ###
    #######################

    V_res = result['x']
    T_res = np.array(V_res[0 : npar]).flatten() # (1, npar)
    u_res = np.array(V_res[npar : npar + solver.Nu*nu]).reshape((solver.Nu, nu)) # (Nu, nu)
    x_res = np.array(V_res[npar + solver.Nu*nu : ]).reshape((solver.Nx, nx)) # (Nx, nx)

    #########################
    ### PREPARE SOLUTIONS ###
    #########################
    #  0, 1, 2,    3, 4
    # [m, v, beta, h, phi]
    # Extend u_res by one at each stage inerface to be same length as x_res
    ins_idx = np.cumsum(solver.N)
    ins_vals = np.vstack((u_res[ins_idx[:-1]], u_res[-1]))
    u_res = np.insert(u_res, ins_idx, ins_vals, axis=0) # u_res should now be size (sum(N+1), nu)

    xp_r = x_res[:, 3] + solver.body.r_0
    xp_theta = np.full((solver.Nx), np.pi/2)
    xp_phi = x_res[:, 4]
    R_sph = np.array([
        [np.sin(xp_theta) * np.cos(xp_phi), np.sin(xp_theta) * np.sin(xp_phi), np.cos(xp_theta)],  # er
        [np.cos(xp_theta) * np.cos(xp_phi), np.cos(xp_theta) * np.sin(xp_phi), -np.sin(xp_theta)], # etheta
        [-np.sin(xp_phi), np.cos(xp_phi), np.zeros(solver.Nx)]                                   # ephi
    ]) # shape is (3, 3, Nx)

    R_sph = np.moveaxis(R_sph, -1, 0) # shape should now be (Nx, 3, 3)

    xp_pos = R_sph[:, 0]*xp_r[:, np.newaxis]

    xp_beta = x_res[:, 2]
    xp_vmag = x_res[:, 1]
    xp_vsph = np.array([xp_vmag*np.cos(xp_beta), np.zeros(solver.Nx), xp_vmag*np.sin(xp_beta)]).T
    R_sph_T = np.transpose(R_sph, axes=(0, 2, 1))
    xp_vel = np.matmul(R_sph_T, xp_vsph[..., np.newaxis]).squeeze(-1)    

    ### NOW GO BACK ###
    x_pos = (R.T @ xp_pos.T).T
    x_vel = (R.T @ xp_vel.T).T

    # Skew if requested
    if opts.get('skew', False):
        print('Skewing start of trajectory.')
        for i in range(0, solver.Nx):
            # Calculate rot_angle based on phi of 2D solution
            angle = (rot_angle-np.pi/2)*(1-(x_res[i, 4] - x_res[0, 4])/(x_res[-1, 4] - x_res[0, 4]))**opts.get('skew_strength', 5)
            x_pos[i] = rotate(vec=x_pos[i], axis=Yp, angle=angle)
            x_vel[i] = rotate(vec=x_vel[i], axis=Yp, angle=angle)

    x_ctrl = np.array([u_res.flatten(), np.arctan2(x_vel[:, 1], x_vel[:, 0]), np.arcsin(-x_vel[:, 2]/xp_vmag)]).T

    sols = []
    for k in range(0, solver.nstages):
        block_start = sum(solver.N[0:k]) + k 
        block_end = sum(solver.N[0:k+1]) + k+1
        m = x_res[block_start : block_end, 0]
        pos = x_pos[block_start: block_end]
        vel = x_vel[block_start: block_end]
        ctrl = x_ctrl[block_start: block_end]
        t, dt = np.linspace(0, T_res[k], solver.N[k]+1, retstep=True)

        sols.append(
            Solution(X=np.block([m[:, np.newaxis], pos, vel]).tolist(),
                U=ctrl.tolist(),
                stage=k+1, 
                t=t.tolist(),
                )
            )
    return {'sols': sols, 
            'status': nlpsolver.stats()['return_status'], 
            'success': nlpsolver.stats()['return_status'] == 'Solve_Succeeded', 
            'iter_count': nlpsolver.stats()['iter_count']}