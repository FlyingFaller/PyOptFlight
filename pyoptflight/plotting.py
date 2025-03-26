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
from .solver import Solver
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
    
    fig = plot_celestial(solver.body,
                         show_stars=False,
                         hpx=hpx,
                         background_color=background_color,
                         show_axis=show_axis,
                         size=size)
    
    if show_target_orbit:
        if solver.config.landing:
            pos = solver.x0.get_x0s(solver, npoints=1)['pos'][0]
            vel = solver.x0.get_x0s(solver, npoints=1)['vel'][0]
        else:
            pos = solver.xf.get_x0s(solver, npoints=1)['pos'][0]
            vel = solver.xf.get_x0s(solver, npoints=1)['vel'][0]
        e, a, i, ω, Ω, ν, h_vec, e_vec = state_to_kep(np.block([pos, vel]), solver.body.mu)
        fig = plot_orbit(fig, e, a, i, ω, Ω, solver.body.mu)



    stage_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    for idx in indices:
        if show_actual_orbit:
            if solver.config.landing:
                pos_vel = solver.sols[idx][0].X[0][1:7]
            else:
                pos_vel = solver.sols[idx][-1].X[-1][1:7]
            e, a, i, ω, Ω, ν, h_vec, e_vec = state_to_kep(np.array(pos_vel), solver.body.mu)
            fig = plot_orbit(fig, e, a, i, ω, Ω, solver.body.mu)

        if colorscale == 'vel':
            cmin = min([np.min(np.sum((np.array(sol.X)[:, 4:7])**2, axis=1)**0.5) for sol in solver.sols[idx]])
            cmax = max([np.max(np.sum((np.array(sol.X)[:, 4:7])**2, axis=1)**0.5) for sol in solver.sols[idx]])
        elif colorscale == 'f':
            cmin = 0
            cmax = 1
        else:
            cmin = 0
            cmax = 1

        for k in range(0, solver.nstages):
            sol = solver.sols[idx][k]
            fig = plot_trajectory(fig, 
                                  np.array(sol.X)[:, 1:4],
                                  np.array(sol.X)[:, 4:7],
                                #   np.array(sol.X)[:, 7:10],
                                  np.array(sol.U),
                                  markers=markers, 
                                  colorscale = colorscale,
                                  color=stage_colors[k%len(stage_colors)],
                                  cmin=cmin,
                                  cmax=cmax)
            
    if show_stars:
        fig = add_background_stars(fig)

    return fig

def plot_controls(solver: Solver, indices=[-1]):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.ion()
    for idx in indices:
        sol_set = solver.sols[idx]
        t_offset = 0
        t_full = np.zeros(1)
        u_full = []
        for k, sol in enumerate(sol_set):
            t = np.array(sol.t[1:]) + t_offset
            t_full = np.concatenate((t_full, t))
            t_offset += sol.t[-1]
            u_full += sol.U
            if k + 1 < solver.nstages:
                ax.axvline(t_offset, color='gray', linestyle='--')

    u_full = np.array(u_full)
    ax.plot(t_full[:-1], u_full[:, 0], label='$f$')
    ax.plot(t_full[:-1], u_full[:, 1], label='$\\psi$')
    ax.plot(t_full[:-1], u_full[:, 2], label='$\\theta$')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Throttle %, Angle [rad]')
    ax.set_title('Control vs Time')
    ax.legend()
    ax.grid(True)

    return fig