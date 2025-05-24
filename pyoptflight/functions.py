import json
import casadi as ca
import numpy as np
import pandas as pd
import os
from typing import Any, Optional, Dict

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

def load_json(filename: str) -> dict[str, Any]:
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

def load_csv(
    filename: str,
    base_path: Optional[str] = None,
    header_row_num: Optional[int] = None,
    comment_char: Optional[str] = '#',
    skip_blank_lines: Optional[bool] = True,
    on_bad_lines: Optional[str] = 'skip',
    **pd_read_csv_kwargs
) -> Optional[Dict[str, Any]]:
    """Loads tables from a CSV"""
    if base_path:
        file_path = os.path.join(base_path, filename)
    else:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, filename)

    # Prepare arguments for pd.read_csv
    kwargs_for_pandas = {
        'header': header_row_num,
        'comment': comment_char,
        'skip_blank_lines': skip_blank_lines,
        'on_bad_lines': on_bad_lines,
        **pd_read_csv_kwargs # User's direct kwargs take precedence
    }
    # Remove None values so pandas uses its defaults if not specified
    kwargs_for_pandas = {k: v for k, v in kwargs_for_pandas.items() if v is not None or k in pd_read_csv_kwargs}

    try:
        df = pd.read_csv(file_path, **kwargs_for_pandas)

        # Not a perfect parsing for every occation but should be manageble by whatever uses this function
        # Though I will say mixed-type numpy arrays are cursed and the used should be careful to extract the actual
        # data out if string labels are being used at the beginning of rows!!
        result: Dict[str, Any] = {
            "header": df.columns.numpy(), 
            "data": df.to_numpy(na_value=np.nan),
        }

        return result

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or resulted in an empty DataFrame after processing.")
        return None
    except Exception as e:
        print(f"An error occurred while parsing CSV file '{file_path}': {e}")
        return None

def kep_to_state(e: float, a: float, i: float, ω: float, Ω: float, ν: float, μ:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def state_to_kep(state_vec: np.ndarray, μ: float) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
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