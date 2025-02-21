import json
import casadi as ca
import numpy as np
import os

class AutoRepr:
    """Automatically generates a string representation of the object."""
    def __repr__(self, indent=0):
        single_indent = "    " * indent
        double_indent = "    " * (indent + 1)
        class_name = self.__class__.__name__

        # Calculate the longest attribute name for alignment
        attr_keys = vars(self).keys()
        max_key_length = max(len(key) for key in attr_keys)

        attributes = []
        for key, value in vars(self).items():
            # Align the '=' based on the longest attribute name
            padding = " " * (max_key_length - len(key))
            if isinstance(value, AutoRepr):  # Nested object
                # Pass `is_nested=True` for nested objects to avoid extra leading spaces
                attributes.append(
                    f"{double_indent}{key}{padding} = {value.__repr__(indent + 1)}"
                )
            else:  # Primitive attribute
                attributes.append(f"{double_indent}{key}{padding} = {value}")

        attributes_str = ",\n".join(attributes)

        return f"{class_name}(\n{attributes_str}\n{single_indent})"

def load_json(filename):
    """Loads a JSON file and returns a dictionary."""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, filename)

    json_dict = {}
    try:
        with open(file_path, "r") as f:
            json_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load defaults from {file_path}: {e}")
    return json_dict

def _old_change_basis(states, controls, old_basis, new_basis):
    """Converts state(s) and control(s) between Cartesian and spherical bases."""
    if old_basis == new_basis:
        return states, controls

    dim = states.ndim
    if dim == 1:
        states = np.array([states])
        controls = np.array([controls])

    new_states = np.empty_like(states, dtype=float)
    new_controls = np.empty_like(controls, dtype=float)
    if old_basis == "cart":
        for i, (state, control) in enumerate(zip(states, controls)):
            x, y, z, vx, vy, vz = state
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z/r)
            phi = np.arctan2(y, x)
            er = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            etheta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
            ephi = np.array([-np.sin(phi), np.cos(phi), 0])
            vel = np.array([vx, vy, vz])
            v_r = np.dot(vel, er)
            v_theta = np.dot(vel, etheta)
            v_phi = np.dot(vel, ephi)
            new_states[i] = np.array([r, theta, phi, v_r, v_theta/r, v_phi/(r*np.sin(theta))])

            fx, fy, fz = control
            f = np.sqrt(fx**2 + fy**2 + fz**2)
            beta = np.arccos(np.dot(er, control)/f)
            control_planar = control - np.dot(er, control)*er
            gamma = np.arctan2(np.dot(np.cross(ephi, control_planar), er), np.dot(ephi, control_planar))
            new_controls[i] = np.array([f, gamma, beta])
            
    elif old_basis == "sph":
        for i, (state, control) in enumerate(zip(states, controls)):
            r, theta, phi, v_r, omega, psi = state
            v_theta = r*omega
            v_phi = r*np.sin(theta)*psi
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            er = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            etheta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
            ephi = np.array([-np.sin(phi), np.cos(phi), 0])
            vel = v_r*er + v_theta*etheta + v_phi*ephi
            new_states[i] = np.array([x, y, z, vel[0], vel[1], vel[2]])

            f, gamma, beta = control
            new_controls[i] = f*np.cos(beta)*er -f*np.sin(beta)*np.sin(gamma)*etheta + f*np.sin(beta)*np.cos(gamma)*ephi

    if dim == 1:
        return new_states[0], new_controls[0]
    else:
        return new_states, new_controls

def change_basis(states, controls, old_basis, new_basis):
    """Converts state(s) and control(s) between Cartesian and spherical bases.
    
    If `controls` is None, only the state conversion is performed.
    """
    if old_basis == new_basis:
        return states if controls is None else (states, controls)

    # Ensure states is at least 2D.
    dim = states.ndim
    states = np.atleast_2d(states)
    if controls is not None:
        controls = np.atleast_2d(controls)
    
    new_states = np.empty_like(states, dtype=float)
    if controls is not None:
        new_controls = np.empty_like(controls, dtype=float)
    
    # Create an iterator for controls: if controls is None, use a list of None.
    control_iter = controls if controls is not None else [None] * len(states)

    if old_basis == "cart":
        for i, (state, control) in enumerate(zip(states, control_iter)):
            x, y, z, vx, vy, vz = state
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            er = np.array([np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(theta)])
            etheta = np.array([np.cos(theta)*np.cos(phi),
                               np.cos(theta)*np.sin(phi),
                               -np.sin(theta)])
            ephi = np.array([-np.sin(phi), np.cos(phi), 0])
            vel = np.array([vx, vy, vz])
            v_r = np.dot(vel, er)
            v_theta = np.dot(vel, etheta)
            v_phi = np.dot(vel, ephi)
            new_states[i] = np.array([r, theta, phi, v_r, v_theta/r, v_phi/(r*np.sin(theta))])
            
            if control is not None:
                # FIXME: Will break if f is 0 better to use R matrix or similar
                fx, fy, fz = control
                f = np.sqrt(fx**2 + fy**2 + fz**2)
                beta = np.arccos(np.dot(er, control) / f)
                control_planar = control - np.dot(er, control) * er
                gamma = np.arctan2(np.dot(np.cross(ephi, control_planar), er),
                                   np.dot(ephi, control_planar))
                new_controls[i] = np.array([f, gamma, beta])
                
    elif old_basis == "sph":
        for i, (state, control) in enumerate(zip(states, control_iter)):
            r, theta, phi, v_r, omega, psi = state
            v_theta = r * omega
            v_phi = r * np.sin(theta) * psi
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            er = np.array([np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(theta)])
            etheta = np.array([np.cos(theta)*np.cos(phi),
                               np.cos(theta)*np.sin(phi),
                               -np.sin(theta)])
            ephi = np.array([-np.sin(phi), np.cos(phi), 0])
            vel = v_r * er + v_theta * etheta + v_phi * ephi
            new_states[i] = np.array([x, y, z, vel[0], vel[1], vel[2]])
            
            if control is not None:
                f, gamma, beta = control
                new_controls[i] = (f * np.cos(beta) * er -
                                   f * np.sin(beta) * np.sin(gamma) * etheta +
                                   f * np.sin(beta) * np.cos(gamma) * ephi)

    # If the input was 1D, return a single state (and control if applicable)
    if dim == 1:
        new_states = new_states[0]
        if controls is not None:
            new_controls = new_controls[0]

    return new_states if controls is None else (new_states, new_controls)

def rotate_trajectory(states, controls, axis, angle, old_basis='sph', new_basis='sph'):
    """Rotates a trajectory about an axis by a given angle."""
    if controls is not None:
        states, controls = change_basis(states, controls, old_basis, "cart")
    else:
        states = change_basis(states, controls, old_basis, "cart")

    dim = states.ndim
    states = np.atleast_2d(states)
    if controls is not None:
        controls = np.atleast_2d(controls)

    new_states = np.empty_like(states, dtype=float)
    if controls is not None:
        new_controls = np.empty_like(controls, dtype=float)

    control_iter = controls if controls is not None else [None] * len(states)

    for i, (state, control) in enumerate(zip(states, control_iter)):
        pos = state[0:3]
        vel = state[3:6]
        new_pos = pos*np.cos(angle) + np.cross(axis, pos)*np.sin(angle) + axis*np.dot(axis, pos)*(1 - np.cos(angle))
        new_vel = vel*np.cos(angle) + np.cross(axis, vel)*np.sin(angle) + axis*np.dot(axis, vel)*(1 - np.cos(angle))
        new_states[i] = np.concatenate((new_pos, new_vel))
        
        if control is not None:
            new_controls[i] = control*np.cos(angle) + np.cross(axis, control)*np.sin(angle) + axis*np.dot(axis, control)*(1 - np.cos(angle))


    if controls is not None:
        new_states, new_controls = change_basis(new_states, new_controls, "cart", new_basis)
    else:
        new_controls = None
        new_states = change_basis(new_states, new_controls, "cart", new_basis)

    if dim == 1:
        new_states = new_states[0]
        if controls is not None:
            new_controls = new_controls[0]
    return new_states if controls is None else (new_states, new_controls)

def kep_to_state(e, a, i, ω, Ω, ν, μ):
    """Converts Keplerian elements to Cartesian state vector."""
    r = a * (1 - e**2) / (1 + e * np.cos(ν))
    p = a * (1 - e**2)
    
    r_p = np.array([r * np.cos(ν), r * np.sin(ν), 0])
    
    v_p = np.array([
        -np.sqrt(μ / p) * np.sin(ν),
        np.sqrt(μ / p) * (e + np.cos(ν)),
        0
    ])
    
    cΩ, sΩ = np.cos(Ω), np.sin(Ω)
    cω, sω = np.cos(ω), np.sin(ω)
    ci, si = np.cos(i), np.sin(i)
    
    Q = np.array([
        [cΩ * cω - sΩ * sω * ci, -cΩ * sω - sΩ * cω * ci, sΩ * si],
        [sΩ * cω + cΩ * sω * ci, -sΩ * sω + cΩ * cω * ci, -cΩ * si],
        [sω * si, cω * si, ci]
    ])
    
    r_vec = Q @ r_p
    v_vec = Q @ v_p

    h_vec = np.cross(r_vec, v_vec)
    e_vec = np.cross(v_vec, h_vec) / μ - r_vec / np.linalg.norm(r_vec)

    return np.concatenate([r_vec, v_vec]), h_vec, e_vec

def state_to_kep(state_vec, μ):
    """Converts Cartesian state vector to Keplerian elements."""
    r_vec = state_vec[0:3]
    r = np.linalg.norm(r_vec)

    v_vec = state_vec[3:6]
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = np.cross(v_vec, h_vec) / μ - r_vec / r
    e = np.linalg.norm(e_vec)

    N_vec = np.cross(np.array([0, 0, 1]), h_vec)
    if np.allclose(N_vec, np.zeros(3)):
        N_vec = np.array([1, 0, 0])
        Ω = 0
    else:
        Ω = np.arctan2(h_vec[0], -h_vec[1]) # LAN

    E = 0.5*v**2 - μ/r # Specific orbital energy
    a = -μ/(2*E) # SMA

    i = np.arccos(h_vec[2]/h) # inclination

    if np.allclose(e_vec, np.zeros(3)):
        ω = 0 # Arg of Peri
        ν = np.arctan2(np.dot(np.cross(N_vec, r_vec), h_vec/h), np.dot(N_vec, r_vec))
    else:
        ω = np.arctan2(np.dot(np.cross(N_vec, e_vec), h_vec/h), np.dot(N_vec, e_vec))
        ν = np.arctan2(np.dot(np.cross(e_vec, r_vec), h_vec/h), np.dot(e_vec, r_vec))
    return e, a, i, ω, Ω, ν, h_vec, e_vec

def sym_state_to_he(state_vec, μ):
    """Converts CasADi symbolic Cartesian state vector to h and e vectors."""
    r, θ, ϕ = state_vec[0], state_vec[1], state_vec[2]
    vr, ω, ψ = state_vec[3], state_vec[4], state_vec[5]
    vel = ca.vertcat(
        ω*r*ca.cos(ϕ)*ca.cos(θ) - ψ*r*ca.sin(ϕ)*ca.sin(θ) + vr*ca.sin(θ)*ca.cos(ϕ),
        ω*r*ca.sin(ϕ)*ca.cos(θ) + ψ*r*ca.cos(ϕ)*ca.sin(θ) + vr*ca.sin(θ)*ca.sin(ϕ),
        -ω*r*ca.sin(θ) + vr*ca.cos(θ)
    )
    er = ca.vertcat(
        ca.sin(θ)*ca.cos(ϕ),
        ca.sin(ϕ)*ca.sin(θ),
        ca.cos(θ)
    )
    h_vec = ca.cross(r*er, vel)
    e_vec = ca.cross(vel, h_vec)/μ - er
    return h_vec, e_vec