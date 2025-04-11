import numpy as np
from numpy import pi, sin, cos
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import skimage.io as sio
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from skimage.transform import resize
from .functions import *
import os
from .solver import Solver, StageSolution, FlightSolution, SolverContext
from .setup import Body
from .boundary_objects import *
from typing import List, Dict

def _image2zvals(img,  n_colors=64, n_training_pixels=800, rngs = 123): 
    # Image color quantization
    # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
    # n_colors: int,  number of colors for color quantization
    # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
    # returns the array of z_values for the heatmap representation, and a plotly colorscale
   
    if img.ndim != 3:
        raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
    rows, cols, d = img.shape
    if d < 3:
        raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
        
    range0 = img[:, :, 0].max() - img[:, :, 0].min()
    if range0 > 1: #normalize the img values
        img = np.clip(img.astype(float)/255, 0, 1)
        
    observations = img[:, :, :3].reshape(rows*cols, 3)
    training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
    model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
    
    codebook = model.cluster_centers_
    indices = model.predict(observations)
    z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
    z_vals = z_vals.reshape(rows, cols)
    # define the Plotly colorscale with n_colors entries    
    scale = np.linspace(0, 1, n_colors)
    colors = (codebook*255).astype(np.uint8)
    pl_colorscale = [[float(sv), f'rgb{tuple([int(c) for c in color])}'] for sv, color in zip(scale, colors)]
      
    # Reshape z_vals  to  img.shape[:2]
    return z_vals.reshape(rows, cols), pl_colorscale

def _regular_tri(rows, cols):
    #define triangles for a np.meshgrid(np.linspace(a, b, cols), np.linspace(c,d, rows))
    triangles = []
    for i in range(rows-1):
        for j in range(cols-1):
            k = j+i*cols
            triangles.extend([[k,  k+cols, k+1+cols], [k, k+1+cols, k+1]])
    return np.array(triangles) 
       
def _mesh_data(img, n_colors=32, n_training_pixels=800):
    rows, cols, _ = img.shape
    z_data, pl_colorscale = _image2zvals(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
    triangles = _regular_tri(rows, cols) 
    I, J, K = triangles.T
    zc = z_data.flatten()[triangles] 
    tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
    return I, J, K, tri_color_intensity, pl_colorscale

def plot_spheres(spheres: List[Dict],
                 show_axis=False,
                 size=(1000, 1000), 
                 ncolors=32, 
                 npixels=10000, 
                 background_color='black', 
                 hpx = 512):
    
    def sphere_mesh(radius, rows, cols):
        u, v = np.meshgrid(np.linspace(-np.pi, np.pi, cols), np.linspace(-pi/2, pi/2, rows))
        return radius*cos(u)*cos(v), radius*sin(u)*cos(v), radius*sin(v)
    
    fig = go.Figure()

    radii = []
    for sphere in spheres:
        radius = sphere['r']
        radii.append(radius)
        image_path = sphere.get('path', None)
        color = sphere.get('color', 'blue')
        alpha = sphere.get('alpha', 1)

        if image_path is not None:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, image_path)
            img = sio.imread(file_path)

            reduction_factor = img.shape[1]/hpx
            if reduction_factor < 1:
                reduction_factor = 1
            new_shape = (img.shape[0] // reduction_factor, img.shape[1] // reduction_factor)
            img = resize(img, new_shape, anti_aliasing=True)

            r, c, _ = img.shape
            x, y, z = sphere_mesh(radius, r, c)
            I, J, K, tri_color_intensity, pl_colorscale = _mesh_data(img, n_colors=ncolors, n_training_pixels=npixels) 

            fig.add_mesh3d(x=x.flatten(), y=y.flatten(), z=np.flipud(z).flatten(),  
                                        i=I, j=J, k=K, intensity=tri_color_intensity, intensitymode="cell", 
                                        colorscale=pl_colorscale, showscale=False)
        else:
            x, y, z = sphere_mesh(radius, 100, 100)
            sphere_surf = go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, color], [1, color]],  # Solid blue color
                showscale=False,
                opacity=alpha  # Hide the colorbar
                )
            fig = go.Figure(data=[sphere_surf])

    fig.update_layout(width=size[0], height=size[1],
                    margin=dict(t=1, r=1, b=1, l=1),
                    paper_bgcolor=background_color,
                    plot_bgcolor=background_color,
                    scene=dict(
                        aspectmode="cube",
                        xaxis_visible=show_axis, 
                        yaxis_visible=show_axis, 
                        zaxis_visible=show_axis,
                        xaxis=dict(backgroundcolor=background_color),
                        yaxis=dict(backgroundcolor=background_color),
                        zaxis=dict(backgroundcolor=background_color),
                        camera=dict(
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=0, z=0),
                            up=dict(x=0, y=0, z=1)
                        )
                        )
                        )
    return fig

def plot_trajectory(fig, pos, vel, ctrl, 
                    markers: str|None = None, freq=20, scale=100, 
                    colorscale: str|None = None, color='blue', cmin = 0, cmax=1):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]
    vmag = np.sum(vel**2, axis=1)**0.5

    f = ctrl[:, 0]
    psi = ctrl[:, 1]
    theta = ctrl[:, 2]

    # Maybe add AoA or Max Q colorscales later ? # TODO
    N = len(x)
    if colorscale == 'f':
        color_dict = dict(color=f, colorscale='viridis', width=7, cmin=0, cmax=1)
        cscale = scale + f
    elif colorscale == 'vel':
        color_dict = dict(color=vmag, colorscale='viridis', width=7, cmin=cmin, cmax=cmax)
        cscale = scale + vmag
    else:
        color_dict = dict(color=color, width=7)
        cscale = scale + 0.5*np.ones(N)

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               mode='lines', 
                               name='Trajectory',
                               line=color_dict))
    
    # Periodically add arrow vectors along the trajectory
    if markers is not None:
        num_steps = max(2, round(N/freq)+1)
        for i in np.linspace(0, N - 2, num_steps, dtype=int):
            if markers == 'ctrl':
                mkr_dir = np.array([
                    np.cos(psi[i])*np.cos(theta[i]),
                    np.sin(psi[i])*np.cos(theta[i]),
                    -np.sin(theta[i])
                ])*cscale[i]
            elif markers == 'vel':
                mkr_dir = vel[i]/vmag[i]*cscale[i]

            # For cones, mimic a solid color by using a constant colorscale.
            if colorscale is None:
                cone_colorscale = [[0, color], [1, color]]
                cone_trace = go.Cone(
                    x=[x[i]],
                    y=[y[i]],
                    z=[z[i]],
                    u=[mkr_dir[0]],
                    v=[mkr_dir[1]],
                    w=[mkr_dir[2]],
                    showscale=False,
                    colorscale=cone_colorscale
                )
            else:
                cone_trace = go.Cone(
                    x=[x[i]],
                    y=[y[i]],
                    z=[z[i]],
                    u=[mkr_dir[0]],
                    v=[mkr_dir[1]],
                    w=[mkr_dir[2]],
                    showscale=False,
                    colorscale='viridis',
                    cmin=scale+cmin,
                    cmax=scale+cmax
                )
            fig.add_trace(cone_trace)
            
    eye_dir = pos[int((N+1)/2)]/np.linalg.norm(pos[int((N+1)/2)])
    eye_theta = np.arccos(eye_dir[2]) - np.pi/4 if np.arccos(eye_dir[2]) - np.pi/4 >= np.pi/4 else np.arccos(eye_dir[2]) + np.pi/4
    eye_phi = np.arctan2(eye_dir[1], eye_dir[0])
    fig.update_traces(showlegend=False)
    fig.update_layout(scene=dict(
                        camera=dict(
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x = 1.5 * np.sin(eye_theta) * np.cos(eye_phi),
                                     y = 1.5 * np.sin(eye_theta) * np.sin(eye_phi),
                                     z = 1.5 * np.cos(eye_theta)),
                            up=dict(x=0, y=0, z=1)
                        )
                        )
                        )
    return fig

def plot_orbit(fig, e, a, i, ω, Ω, μ):
    ν_range = np.linspace(0, 2*np.pi, 1000)
    positions = np.zeros((1000, 3))
    for k, ν in enumerate(ν_range):
        state, _, _ = kep_to_state(e, a, i, ω, Ω, ν, μ)
        positions[k] = state[0:3]
    fig.add_trace(go.Scatter3d(x=positions[:, 0], 
                               y=positions[:, 1], 
                               z=positions[:, 2], 
                                mode='lines', 
                                name='Orbit',
                                line=dict(width=4)))
    fig.update_traces(showlegend=False)
    return fig
    
def plot_points(fig, points, color, label='', size=10, legend=False, focus=False):
    if points.ndim == 1:
        points  = [points]
    fig.add_trace(go.Scatter3d(x=points[:, 0], 
                               y=points[:, 1], 
                               z=points[:, 2], 
                               mode='markers', 
                               name=label, 
                               marker=dict(color=color, size=size)
                               )
                    )
    fig.update_traces(showlegend=legend)
    if focus:
        eye_point = points[0]/np.linalg.norm(points[0])
        fig.update_layout(scene=dict(
                        camera=dict(
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x = 1.5*eye_point[0],
                                     y = 1.5*eye_point[1],
                                     z = 1.5*eye_point[2]),
                            up=dict(x=0, y=0, z=1)
                            )
                        )
                        )
    return fig

def add_background_stars(fig, skybox_rad=None, nstars_per_face=500,
                         star_size=1, star_color='white'):
    """
    Add background stars to a 3D Plotly figure.

    This function creates random white dot "stars" on the six faces of a cube
    centered at (0,0,0). By choosing the cube size (via star_box_distance) to be 
    much larger than your planet, the stars are always in the background – they 
    will not come between the camera and the planet when you are zoomed in.

    If star_box_distance is None, the function attempts to automatically determine
    a reasonable distance by examining the current scene ranges (or, if necessary, 
    the data). It then sets star_box_distance to 3 times the maximum absolute value 
    among the x, y, and z ranges.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to which the stars will be added.
    n_stars_per_face : int, optional
        Number of stars to generate on each face of the cube (default is 100).
    star_box_distance : float, optional
        The absolute coordinate value for each face of the cube. The cube spans
        from -star_box_distance to +star_box_distance along x, y, and z. If None,
        the function will compute a value based on the figure's data.
    star_marker_size : int or float, optional
        The marker size for each star (default is 2).
    star_color : str, optional
        The marker color for the stars (default is 'white').

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The same figure, with a new trace added for the stars.
    """

    # Determine extend of data
    x_min_nonpad = min([np.min(obj.x) for obj in fig.data])
    x_max_nonpad = max([np.max(obj.x) for obj in fig.data])
    y_min_nonpad = min([np.min(obj.y) for obj in fig.data])
    y_max_nonpad = max([np.max(obj.y) for obj in fig.data])
    z_min_nonpad = min([np.min(obj.z) for obj in fig.data])
    z_max_nonpad = max([np.max(obj.z) for obj in fig.data])

    x_min = x_min_nonpad - 0.05*(x_max_nonpad - x_min_nonpad)
    x_max = x_max_nonpad + 0.05*(x_max_nonpad - x_min_nonpad)
    y_min = y_min_nonpad - 0.05*(y_max_nonpad - y_min_nonpad)
    y_max = y_max_nonpad + 0.05*(y_max_nonpad - y_min_nonpad)
    z_min = z_min_nonpad - 0.05*(z_max_nonpad - z_min_nonpad)
    z_max = z_max_nonpad + 0.05*(z_max_nonpad - z_min_nonpad)

    # Capture camera eye location
    camera_eye = fig.layout.scene.camera.eye

    # Compute bounding box center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2

    # Compute bounding box size
    scale_x = x_max - x_min
    scale_y = y_max - y_min
    scale_z = z_max - z_min

    # Convert to data coordinates
    data_x = center_x + camera_eye.x * scale_x
    data_y = center_y + camera_eye.y * scale_y
    data_z = center_z + camera_eye.z * scale_z

    # If skybox_rad is not provided autoscale
    if skybox_rad is None:
        skybox_rad = 5*max(abs(x_min_nonpad), abs(x_max_nonpad),
                           abs(y_min_nonpad), abs(y_max_nonpad),
                           abs(z_min_nonpad), abs(z_max_nonpad))

    # We'll accumulate star coordinates for each face of a cube with half-length star_box_distance.
    xs = []
    ys = []
    zs = []

    # Face: x = +star_box_distance (right face)
    xs.append(np.full(nstars_per_face, skybox_rad))
    ys.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    zs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))

    # Face: x = -star_box_distance (left face)
    xs.append(np.full(nstars_per_face, -skybox_rad))
    ys.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    zs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))

    # Face: y = +star_box_distance (front face)
    xs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    ys.append(np.full(nstars_per_face, skybox_rad))
    zs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))

    # Face: y = -star_box_distance (back face)
    xs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    ys.append(np.full(nstars_per_face, -skybox_rad))
    zs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))

    # Face: z = +star_box_distance (top face)
    xs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    ys.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    zs.append(np.full(nstars_per_face, skybox_rad))

    # Face: z = -star_box_distance (bottom face)
    xs.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    ys.append(np.random.uniform(-skybox_rad, skybox_rad, nstars_per_face))
    zs.append(np.full(nstars_per_face, -skybox_rad))

    # Concatenate all points together
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)

    # Create a 3D scatter trace for the stars.
    star_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=star_size,
            color=star_color,
        ),
        hoverinfo='none',
        showlegend=False,
    )

    fig.add_trace(star_trace)

    ex = (data_x - 0) / (2*skybox_rad)
    ey = (data_y - 0) / (2*skybox_rad)
    ez = (data_z - 0) / (2*skybox_rad)

    fig.update_layout(scene=dict(
                        camera=dict(
                            eye=dict(x = ex,
                                     y = ey,
                                     z = ez),
                                     )
                                    )
                                )

    return fig

def plot_celestial(body: Body, 
                   show_stars=True,  
                   hpx=512, 
                   background_color='black', 
                   show_axis=False, 
                   size=(1000, 1000)):
    
    body_dict = {'r': body.r_0, 
                 'path': body.meshpath}
    
    if body.atm is not None:
        atm_dict = {'r': body.r_0+body.atm.cutoff_altitude,
            'color': body.atm.color,
            'alpha': 0.1}
        spheres = [atm_dict, body_dict]
    else:
        spheres = [body_dict]

    fig = plot_spheres(spheres, show_axis=show_axis, size=size, background_color=background_color, hpx=hpx)
    if show_stars:
        fig = add_background_stars(fig)
    return fig

def plot_solutions(solver: Solver, 
                   indices = [-1], 
                   show_stars=True, 
                   show_target_orbit=False,
                   show_actual_orbit=False,
                   markers=None, 
                   colorscale=None, 
                   hpx=512, 
                   background_color='black', 
                   show_axis=False, 
                   size=(1000, 1000)):
    
    fig = plot_celestial(solver.context.body,
                         show_stars=False,
                         hpx=hpx,
                         background_color=background_color,
                         show_axis=show_axis,
                         size=size)
    
    if show_target_orbit:
        if solver.context.config.landing:
            pos = solver.x0.get_x0s(solver.context, npoints=1)['pos'][0]
            vel = solver.x0.get_x0s(solver.context, npoints=1)['vel'][0]
        else:
            pos = solver.xf.get_x0s(solver.context, npoints=1)['pos'][0]
            vel = solver.xf.get_x0s(solver.context, npoints=1)['vel'][0]
        e, a, i, ω, Ω, ν, h_vec, e_vec = state_to_kep(np.block([pos, vel]), solver.context.body.mu)
        fig = plot_orbit(fig, e, a, i, ω, Ω, solver.context.body.mu)



    stage_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    for idx in indices:
        if show_actual_orbit:
            if solver.context.config.landing:
                pos_vel = solver.stage_sols[idx][0].X[0][1:7]
            else:
                pos_vel = solver.stage_sols[idx][-1].X[-1][1:7]
            e, a, i, ω, Ω, ν, h_vec, e_vec = state_to_kep(np.array(pos_vel), solver.context.body.mu)
            fig = plot_orbit(fig, e, a, i, ω, Ω, solver.context.body.mu)

        if colorscale == 'vel':
            cmin = min([np.min(np.sum((np.array(sol.X)[:, 4:7])**2, axis=1)**0.5) for sol in solver.stage_sols[idx]])
            cmax = max([np.max(np.sum((np.array(sol.X)[:, 4:7])**2, axis=1)**0.5) for sol in solver.stage_sols[idx]])
        elif colorscale == 'f':
            cmin = 0
            cmax = 1
        else:
            cmin = 0
            cmax = 1

        for k in range(0, solver.context.nstages):
            sol = solver.stage_sols[idx][k]
            f_min_constr = solver.constraints[k].f_min
            f_min = f_min_constr.value if f_min_constr.enabled and f_min_constr.value is not None else 0

            U = np.array(sol.U)
            U[:, 0] = U[:, 0] - U[:, 0]*np.maximum(0, np.minimum(1, (f_min - U[:, 0])/solver.context.delta))

            fig = plot_trajectory(fig, 
                                  np.array(sol.X)[:, 1:4],
                                  np.array(sol.X)[:, 4:7],
                                #   np.array(sol.X)[:, 7:10],
                                  U,
                                  markers=markers, 
                                  colorscale = colorscale,
                                  color=stage_colors[k%len(stage_colors)],
                                  cmin=cmin,
                                  cmax=cmax)
            
    if show_stars:
        fig = add_background_stars(fig)

    return fig

def plot_flight_data(solver: Solver, plot_on_nodes=True, use_radians=True, index=-1) -> plt.Figure:
    def split_segments(x, y):
        segments = []
        current_x, current_y = [], []
        
        for xi, yi in zip(x, y):
            if yi is not None:
                current_x.append(xi)
                current_y.append(yi)
            else:
                # if current segment is non-empty, append it as a tuple
                if current_x:
                    segments.append((current_x, current_y))
                    current_x, current_y = [], []
        # Append any segment that might be left over at the end.
        if current_x:
            segments.append((current_x, current_y))
        return segments

    data = solver.flight_sols[index]
    stage_bound = np.cumsum(solver.context.N)[:-1] if plot_on_nodes else np.cumsum([sol.T for sol in solver.stage_sols[index]])[:-1]
    x_label = 'Nodes' if plot_on_nodes else 'Time [s]'

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20, 12))
    axes = axes.flatten()

    # plot stage boundaries on all subplots
    for i, ax in enumerate(axes):
        for bound in stage_bound:
            ax.axvline(bound, linestyle='--', color='gray')

    # 1: mass
    x0 = data.mass.nodes if plot_on_nodes else data.mass.times
    axes[0].plot(x0, data.mass.data, label='Mass', color='blue')
    axes[0].set_title('Mass')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Mass [tons]')
    # axes[0].legend()
    axes[0].grid(True)

    # 2: f, f_eff, f_min
    x1 = data.f.nodes if plot_on_nodes else data.f.times
    f_min = split_segments(x=x1, y=data.f.constraint)
    for i, (x, y) in enumerate(f_min):
        axes[1].plot(x, y, color='red', linestyle='-.')
    axes[1].plot(x1, data.f.data, label='$f$', color='blue')
    axes[1].plot(x1, data.f_eff.data, label='$f_{eff}$', linestyle=':', color='green')
    axes[1].set_title('Throttle')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Throttle [%]')
    axes[1].legend()
    axes[1].grid(True)

    # 3: F_max
    x2 = data.F_max.nodes if plot_on_nodes else data.F_max.times
    axes[2].plot(x2, data.F_max.data, label='$F_{max}$', color='blue')
    axes[2].set_title('Max Thrust')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel('Max Thrust [MN]')
    # axes[2].legend()
    axes[2].grid(True)

    # 4: Isp
    x3 = data.Isp.nodes if plot_on_nodes else data.Isp.times
    axes[3].plot(x3, data.Isp.data, label='$Isp$', color='blue')
    axes[3].set_title('Specific Impulse')
    axes[3].set_xlabel(x_label)
    axes[3].set_ylabel('Specific Impulse [s]')
    # axes[3].legend()
    axes[3].grid(True)

    # 5: rho
    x4 = data.rho.nodes if plot_on_nodes else data.rho.times
    axes[4].plot(x4, data.rho.data, label='$\\rho$', color='blue')
    axes[4].set_title('Density')
    axes[4].set_xlabel(x_label)
    axes[4].set_ylabel('Density [kg/m$^3$]')
    # axes[4].legend()
    axes[4].grid(True)

    # 6: q, q_max
    x5 = data.f.nodes if plot_on_nodes else data.f.times
    q_max = split_segments(x=x5, y=data.q.constraint)
    for i, (x, y) in enumerate(q_max):
        axes[5].plot(x, y, color='red', linestyle='-.')
    axes[5].plot(x5, data.f.data, label='$q$', color='blue')
    axes[5].set_title('Dynamic Pressure')
    axes[5].set_xlabel(x_label)
    axes[5].set_ylabel('Dynamic Pressure [MPa]')
    # axes[5].legend()
    axes[5].grid(True)

    # 7: alpha, alpha_max
    x6 = data.alpha.nodes if plot_on_nodes else data.alpha.times
    alpha_max = split_segments(x=x6, y=data.alpha.constraint)
    for i, (x, y) in enumerate(alpha_max):
        if use_radians:
            axes[6].plot(x, np.array(y), color='red', linestyle='-.')
            axes[6].plot(x, -np.array(y), color='red', linestyle='-.')
        else:
            axes[6].plot(x, np.rad2deg(y), color='red', linestyle='-.')
            axes[6].plot(x, -np.rad2deg(y), color='red', linestyle='-.')
    if use_radians:
        axes[6].plot(x6, data.alpha.data, label='$\\alpha$', color='blue')
    else:
        axes[6].plot(x6, np.rad2deg(data.alpha.data), label='$\\alpha$', color='blue')
    axes[6].set_title('Angle of Attack')
    axes[6].set_xlabel(x_label)
    if use_radians:
        axes[6].set_ylabel('Angle of Attack [rad]')
    else:
        axes[6].set_ylabel('Angle of Attack [deg]')
    # axes[6].legend()
    axes[6].grid(True)

    # 8: tau, tau_max
    x7 = data.tau.nodes if plot_on_nodes else data.tau.times
    tau_max = split_segments(x=x7, y=data.tau.constraint)
    for i, (x, y) in enumerate(tau_max):
        axes[7].plot(x, np.array(y), color='red', linestyle='-.')
        axes[7].plot(x, -np.array(y), color='red', linestyle='-.')
    axes[7].plot(x7, data.tau.data, label='$\\tau$', color='blue')
    axes[7].set_title('Throttle Rate')
    axes[7].set_xlabel(x_label)
    axes[7].set_ylabel('Throttle Rate [%/s]')
    # axes[7].legend()
    axes[7].grid(True)

    # 9: body_rate_y, body_rate_z, max_body_rate_y, max_body_rate_z
    x8a = data.body_rate_y.nodes if plot_on_nodes else data.body_rate_y.times
    x8b = data.body_rate_z.nodes if plot_on_nodes else data.body_rate_z.times
    max_body_rate_y = split_segments(x=x8a, y=data.body_rate_y.constraint)
    max_body_rate_z = split_segments(x=x8b, y=data.body_rate_z.constraint)
    for i, (x, y) in enumerate(max_body_rate_y):
        if use_radians:
            axes[8].plot(x, np.array(y), color='red', linestyle='-.')
            axes[8].plot(x, -np.array(y), color='red', linestyle='-.')
        else:
            axes[8].plot(x, np.rad2deg(y), color='red', linestyle='-.')
            axes[8].plot(x, -np.rad2deg(y), color='red', linestyle='-.')

    for i, (x, y) in enumerate(max_body_rate_z):
        if use_radians:
            axes[8].plot(x, np.array(y), color='orange', linestyle='-.')
            axes[8].plot(x, -np.array(y), color='orange', linestyle='-.')
        else:
            axes[8].plot(x, np.rad2deg(y), color='orange', linestyle='-.')
            axes[8].plot(x, -np.rad2deg(y), color='orange', linestyle='-.')

    axes[8].plot(x8a, data.body_rate_y.data, label='Body Rate $y$ ($q$)', color='blue')
    axes[8].plot(x8b, data.body_rate_z.data, label='Body Rate $z$ ($r$)', color='green')
    axes[8].set_title('Body Angular Rates')
    axes[8].set_xlabel(x_label)
    if use_radians:
        axes[8].set_ylabel('Body Angular [rad/s]')
    else:
        axes[8].set_ylabel('Body Angular [deg/s]')
    axes[8].legend()
    axes[8].grid(True)

    # 10: psi, theta
    x9a = data.psi.nodes if plot_on_nodes else data.psi.times
    x9b = data.theta.nodes if plot_on_nodes else data.theta.times
    if use_radians:
        axes[9].plot(x9a, data.psi.data, label='$\\psi$', color='blue')
        axes[9].plot(x9b, data.theta.data, label='$\\theta$', color='green')
    else:
        axes[9].plot(x9a, np.rad2deg(data.psi.data), label='$\\psi$', color='blue')
        axes[9].plot(x9b, np.rad2deg(data.theta.data), label='$\\theta$', color='green')
    axes[9].set_title('Attitude')
    axes[9].set_xlabel(x_label)
    if use_radians:
        axes[9].set_ylabel('Attitude [rad]')
    else:
        axes[9].set_ylabel('Attitude [deg]')
    axes[9].legend()
    axes[9].grid(True)

    # 11: h
    x10 = data.h.nodes if plot_on_nodes else data.h.times
    axes[10].plot(x10, data.h.data, label='$h$', color='blue')
    axes[10].set_title('Altitude')
    axes[10].set_xlabel(x_label)
    axes[10].set_ylabel('Altitude [km]')
    # axes[10].legend()
    axes[10].grid(True)

    # 12: v_mag
    x11 = data.vel.nodes if plot_on_nodes else data.vel.times
    v_mag = np.sum(data.vel.data**2, axis=1)**0.5
    axes[11].plot(x11, v_mag, label='$\\vert\\vert \\vec{v} \\vert\\vert$', color='blue')
    axes[11].set_title('Velocity Magnitude')
    axes[11].set_xlabel(x_label)
    axes[11].set_ylabel('Velocity Magnitude [km/s]')
    # axes[11].legend()
    axes[11].grid(True)
    # possible plots:
    # 1: mass
    # 2: f, f_eff, f_min
    # 3: F_max
    # 4: Isp
    # 5: rho
    # 6: q, q_max
    # 7: alpha, alpha_max
    # 8: tau, tau_max
    # 9: body rates, body rates max
    # 10: psi, theta
    # 11: h
    # 12: v_mag
    fig.tight_layout()
    return fig