import threading
import time
import numpy as np
import json
import random
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation as R

def start_streams(session_info, stream_dict=None):
    (conn, vessel, refframe) = session_info

    # Define a dictionary of attributes and methods required for each stream
    if stream_dict is None:
        # Default data collection
        stream_dict = {
            'ut': (getattr, conn.space_center, 'ut'),
            'position': (vessel.position, refframe),
            'velocity': (vessel.velocity, refframe),
            'direction': (vessel.direction, refframe),
            'thrust': (getattr, vessel, 'thrust'),
            'mass': (getattr, vessel, 'mass'),
            'mean_altitude': (getattr, vessel.flight(), 'mean_altitude'),
            'pitch': (getattr, vessel.flight(), 'pitch'),
            'heading': (getattr, vessel.flight(), 'heading'),
            'angle_of_attack': (getattr, vessel.flight(), 'angle_of_attack'),
            'sideslip_angle': (getattr, vessel.flight(), 'sideslip_angle'),
            'specific_impulse': (getattr, vessel, 'specific_impulse'),
            'aerodynamic_force': (getattr, vessel.flight(refframe), 'aerodynamic_force'),

            # Orbit streams
            'apoapsis': (getattr, vessel.orbit, 'apoapsis'),
            'periapsis': (getattr, vessel.orbit, 'periapsis'),
            'semi_major_axis': (getattr, vessel.orbit, 'semi_major_axis'),
            'semi_minor_axis': (getattr, vessel.orbit, 'semi_minor_axis'),
            'eccentricity': (getattr, vessel.orbit, 'eccentricity'),
            'inclination': (getattr, vessel.orbit, 'inclination'),
            'longitude_of_ascending_node': (getattr, vessel.orbit, 'longitude_of_ascending_node'),
            'argument_of_periapsis': (getattr, vessel.orbit, 'argument_of_periapsis'),
            'mean_anomaly': (getattr, vessel.orbit, 'mean_anomaly'),
            'true_anomaly': (getattr, vessel.orbit, 'true_anomaly')
        }

    # Dynamically initialize streams
    streams = {key: conn.add_stream(*spec) for key, spec in stream_dict.items()}
    return streams

def stop_streams(streams):
    # Clean up streams
    for stream in streams.values():
        stream.remove()

class Recording:
    def __init__(self, session_info, streams, variables=None, game_time = False, auto_stage=False, stages_to_go=0):
        (conn, vessel, refframe) = session_info
        # Initialize recorded data and interval for automatic recording
        if variables is None:
            self.data = {key: [] for key in streams.keys()}
        else:
            self.data = {key: [] for key in variables}
        
        self.streams = streams
        self._recording = False
        self._recording_thread = None
        self.auto_stage = auto_stage
        self.stages_to_go = stages_to_go
        self.conn = conn
        self.vessel = vessel
        self.refframe = refframe
        self.paused = conn.add_stream(getattr, conn.krpc, 'paused')
        self.warp_rate = conn.add_stream(getattr, conn.space_center, 'warp_rate')
        self.game_time = game_time

    def record_point(self):
        """Records a single point to the data list."""
        if self.streams['thrust']() == 0 and self.stages_to_go > 0 and self.auto_stage:
            self.vessel.control.activate_next_stage()
            self.stages_to_go -= 1

        for key in self.data.keys():
            self.data[key].append(self.streams[key]())

    def start_recording(self, dt):
        """Starts automated recording in a separate thread."""
        if not self._recording:
            self.dt = dt
            self._recording = True
            self._recording_thread = threading.Thread(target=self._auto_record)
            self._recording_thread.start()
            print("Automated recording started.")

    def _auto_record(self):
        """Records data at preset intervals."""
        while self._recording:
            if not self.paused():
                self.record_point()
            if self.game_time:
                dt = self.dt/self.warp_rate()
            else:
                dt = self.dt
            time.sleep(dt)

    def stop_recording(self):
        """Stops automated recording."""
        if self._recording:
            self._recording = False
            self._recording_thread.join()
            print("Automated recording stopped.")


    def save_recording(self, filename="recording", format="npz"):
        """
        Saves the recorded data to a file in the specified format (npz or json).

        Parameters:
            filename (str): Base name of the file to save.
            format (str): Format to save the file in, either 'npz' or 'json'.
        """
        if self._recording:
            self.stop_recording()
        
        self.warp_rate.remove()
        self.paused.remove()

        if format == "npz":
            np.savez(filename + '.npz', **{key: np.array(self.data[key]) for key in self.streams.keys()})
            print(f"Recording saved to {filename}.npz")
        elif format == "json":
            with open(filename + '.json', 'w') as json_file:
                # Convert numpy arrays to lists for JSON serialization
                json_data = {key: np.array(self.data[key]).tolist() for key in self.streams.keys()}
                # json.dump(json_data, json_file, indent=4) # Pretty printing if it needs to be uber human readable
                json.dump(json_data, json_file)
            print(f"Recording saved to {filename}.json")
        else:
            raise ValueError("Unsupported format. Please choose 'npz' or 'json'.")

class OldVectorDrawer:

    def __init__(self, session_info, streams, scale=1, normalize=False, vector_computations=None):
        """
        :param session_info: Tuple containing session connection, vessel, and reference frame.
        :param streams: Dictionary of data streams.
        :param scale: Global scaling factor for vector lengths.
        :param normalize: Whether to normalize all vectors globally.
        :param vector_computations: Dictionary mapping vector names to computation functions and options.
        """
        (self.conn, self.vessel, self.refframe) = session_info
        self.streams = streams
        self.global_scale = scale
        self.global_normalize = normalize
        self.vector_computations = vector_computations or {}
        self._updating = False
        self._update_thread = None
        self.xkcd_colors = list(mcolors.XKCD_COLORS.values())  # Store all XKCD colors as a list
        self.vessel_refframe = self.vessel.reference_frame
        self.text_refframe = self.conn.space_center.ReferenceFrame.create_hybrid(self.vessel_refframe, self.refframe, self.vessel_refframe, self.refframe)

    def register_vector(self, name, compute_function, color=None, scale=None, normalize=None, label=None, label_offset=None):
        """
        Registers a new vector with its computation function and options.
        :param name: Name of the vector.
        :param compute_function: A function that computes the vector(s).
        :param color: A single color tuple or a list of colors for each vector (if multiple vectors are returned).
        :param scale: Scale factor for this vector (overrides global scale).
        :param normalize: Whether to normalize this vector (overrides global normalization).
        :param label: Label to display next to the vector.
        :param label_offset: Distance to place the label from the CoM in the direction of the vector. Placed at end of vector if None.
        """
        self.vector_computations[name] = {
            "function": compute_function,
            "color": color or self._get_random_color(),
            "scale": scale,
            "normalize": normalize,
            "label": label,
            "label_offset": label_offset
        }

    def draw_vectors(self, selected_vectors=None):
        """
        Draws the specified vectors or all registered vectors if none are specified.
        :param selected_vectors: List of vector names to draw.
        """
        # self.conn.drawing.clear()  # Clear previous vectors before updating

        vectors_to_draw = selected_vectors or self.vector_computations.keys()
        for name in vectors_to_draw:
            if name in self.vector_computations:
                vector_data = self.vector_computations[name]
                vectors = vector_data["function"]()  # Compute vector(s)
                color = vector_data.get("color", self._get_random_color())
                scale = vector_data.get("scale", self.global_scale)
                normalize = vector_data.get("normalize", self.global_normalize)
                label = vector_data.get("label", None)
                label_offset = vector_data.get("label_offset", None)
                norm_vectors = self._normalize_vectors(vectors)

                if normalize:
                    draw_vectors = norm_vectors*scale
                else:
                    draw_vectors = vectors*scale

                if "line_objs" in vector_data:
                    if draw_vectors.ndim == 1:  # Single vector
                        vector_data["line_objs"].end = tuple(draw_vectors)
                    elif draw_vectors.ndim == 2:  # Multiple vectors
                        for i, line in enumerate(vector_data["line_objs"]):
                            line.end = tuple(draw_vectors[i])
                else:
                    if draw_vectors.ndim == 1:  # Single vector
                        vector_data["line_objs"] = self._draw_single_vector(draw_vectors, color)
                    elif draw_vectors.ndim == 2:  # Multiple vectors
                        vector_data["line_objs"] = []
                        for i, vector in enumerate(draw_vectors):
                            vector_data["line_objs"].append(self._draw_single_vector(vector, color[i] if isinstance(color, list) else color))

                if label is not None:
                    if label_offset is not None:
                        label_poses = label_offset*norm_vectors
                    else:
                        label_poses = draw_vectors

                    if "text_objs" in vector_data:
                        if label_poses.ndim == 1:  # Single vector
                            vector_data["text_objs"].position = tuple(label_poses)
                        elif label_poses.ndim == 2:  # Multiple vectors
                            for i, text in enumerate(vector_data["text_objs"]):
                                text.position = tuple(label_poses[i])
                    else:
                        if label_poses.ndim == 1:  # Single vector
                            vector_data["text_objs"] = self._draw_single_text(label, label_poses, color)
                        elif label_poses.ndim == 2:  # Multiple vectors
                            vector_data["text_objs"] = []
                            for i, label_pos in enumerate(label_poses):
                                vector_data["text_objs"].append(self._draw_single_text(label[i] if isinstance(label, list) else label, label_pos, color[i] if isinstance(color, list) else color))

    def _draw_single_vector(self, vector, color):
        """
        Draws a single vector with the specified color.
        :param vector: The vector to draw.
        :param color: The color of the vector.
        """
        line = self.conn.drawing.add_direction_from_com(vector, self.refframe, length=1)
        line.color = color
        return line

    def _draw_single_text(self, label, label_pos, color):
        """
        Draws a single text label at the specified position and color.
        :param label: The text label to draw.
        :param label_pos: The position of the label.
        :param color: The color of the label.
        """
        text = self.conn.drawing.add_text(text=label, reference_frame=self.text_refframe, position=label_pos, rotation=(1, 0, 0, 0))
        text.color = color
        return text

    def _normalize_vectors(self, vectors):
        """
        Normalizes vectors to unit length.
        :param vectors: A single vector (1D) or multiple vectors (2D array).
        :return: Normalized vectors.
        """
        if vectors.ndim == 1:  # Single vector
            return vectors / np.linalg.norm(vectors)
        elif vectors.ndim == 2:  # Multiple vectors
            return np.array([vec / np.linalg.norm(vec) for vec in vectors])

    def _get_random_color(self):
        """Returns a random color from the XKCD color list."""
        hex_color = random.choice(self.xkcd_colors)  # Pick a random hex color
        return mcolors.to_rgb(hex_color)  # Convert hex to an RGB tuple

    def clear_vectors(self):
        """Clears all currently drawn vectors."""
        self.conn.drawing.clear()
        for vector_data in self.vector_computations.values():
            vector_data.pop("line_objs", None)
            vector_data.pop("text_objs", None)

    def start_drawing(self, dt=0.1):
        """Starts automatic updating of vectors in a separate thread with interval dt."""
        if not self._updating:
            self._updating = True
            self._update_thread = threading.Thread(target=self._auto_update, args=(dt,))
            self._update_thread.start()
            print("Automatic vector updating started.")

    def _auto_update(self, dt):
        """Runs draw_vectors at specified intervals until stopped."""
        while self._updating:
            self.draw_vectors()
            time.sleep(dt)

    def stop_drawing(self):
        """Stops the automatic updating thread and clears vectors."""
        if self._updating:
            self._updating = False
            self._update_thread.join()
            self.clear_vectors()  # Clear vectors on stop
            print("Automatic vector updating stopped and vectors cleared.")


class VectorDrawer:

    def __init__(self, session_info, streams, scale=1, normalize=False, vector_funcs=None):
        """
        :param session_info: Tuple containing session connection, vessel, and reference frame.
        :param streams: Dictionary of data streams.
        :param scale: Global scaling factor for vector lengths.
        :param normalize: Whether to normalize all vectors globally.
        :param vector_computations: Dictionary mapping vector names to computation functions and options.
        """
        (self.conn, self.vessel, self.refframe) = session_info
        self.streams = streams
        self.global_scale = scale
        self.global_normalize = normalize
        self.vector_funcs = vector_funcs or {}
        self._updating = False
        self._update_thread = None
        self.xkcd_colors = list(mcolors.XKCD_COLORS.values())  # Store all XKCD colors as a list
        self.vessel_refframe = self.vessel.reference_frame
        self.text_refframe = self.conn.space_center.ReferenceFrame.create_hybrid(self.vessel_refframe, self.refframe, self.vessel_refframe, self.refframe)

    def register_vector(self, name, vector_func, colors=None, scale=None, normalize=None, labels=None, label_offset=None):
        """
        Registers a new vector with its computation function and options.
        :param name: Name of the vector.
        :param compute_function: A function that computes the vector(s).
        :param color: A single color tuple or a list of colors for each vector (if multiple vectors are returned).
        :param scale: Scale factor for this vector (overrides global scale).
        :param normalize: Whether to normalize this vector (overrides global normalization).
        :param label: Label to display next to the vector.
        :param label_offset: Distance to place the label from the CoM in the direction of the vector. Placed at end of vector if None.
        """
        self.vector_funcs[name] = {
            "functions": vector_func,
            "colors": colors,
            "scale": scale if scale is not None else self.global_scale,
            "normalize": normalize if normalize is not None else self.global_normalize,
            "labels": labels,
            "label_offset": label_offset,
            "line_objs": None,
            "text_objs": None
        }

    def draw_vectors(self, selected_vectors=None):
        """
        Draws the specified vectors or all registered vectors if none are specified.
        :param selected_vectors: List of vector names to draw.
        """
        vectors_to_draw = selected_vectors or self.vector_funcs.keys()
        for name in vectors_to_draw:
            if name in self.vector_funcs:
                vector_data = self.vector_funcs[name]
                vectors = vector_data["functions"]()  # Compute vector(s)                
                vectors = np.array([vectors]) if vectors.ndim == 1 else np.array(vectors) # Force 2D
                num_vecs = len(vectors)

                colors = vector_data.get("colors", None)
                colors = colors if colors is not None else self._get_random_color(num_vecs) # Generate random colors if needed
                colors = colors if isinstance(colors, list) else [colors] # Force list
                colors = colors if len(colors) == num_vecs else [colors[0] for _ in range(num_vecs)]

                scale = vector_data.get("scale", self.global_scale)
                normalize = vector_data.get("normalize", self.global_normalize)

                labels = vector_data.get("labels", None)

                label_offset = vector_data.get("label_offset", None)
                norm_vectors = self._normalize_vectors(vectors) 

                if normalize:
                    draw_vectors = norm_vectors*scale
                else:
                    draw_vectors = vectors*scale

                if vector_data.get("line_objs", None) is not None:
                    for i, line in enumerate(vector_data["line_objs"]):
                        line.end = tuple(draw_vectors[i])
                else:
                    vector_data["line_objs"] = []
                    for i, vector in enumerate(draw_vectors):
                        vector_data["line_objs"].append(self._draw_single_vector(vector, colors[i]))

                if labels is not None:
                    labels = labels if isinstance(labels, list) else [labels] # Force list
                    labels = labels if len(labels) == num_vecs else [labels[0] for _ in range(num_vecs)]
                    if label_offset is not None:
                        label_poses = label_offset*norm_vectors
                    else:
                        label_poses = draw_vectors

                    if vector_data.get("text_objs", None) is not None:
                        for i, text in enumerate(vector_data["text_objs"]):
                            rotation = self._compute_text_rotation(norm_vectors[i], [0, 0, 0])
                            text.position = tuple(label_poses[i])
                            text.rotation = rotation
                    else:
                        vector_data["text_objs"] = []
                        for i, label_pos in enumerate(label_poses):
                            rotation = self._compute_text_rotation(norm_vectors[i], [0, 0, 0])
                            vector_data["text_objs"].append(self._draw_single_text(labels[i], label_pos, colors[i], rot=rotation))

    def _draw_single_vector(self, vector, color):
        """
        Draws a single vector with the specified color.
        :param vector: The vector to draw.
        :param color: The color of the vector.
        """
        line = self.conn.drawing.add_direction_from_com(vector, self.refframe, length=1)
        line.color = color
        return line

    def _draw_single_text(self, label, label_pos, color, rot):
        """
        Draws a single text label at the specified position and color.
        :param label: The text label to draw.
        :param label_pos: The position of the label.
        :param color: The color of the label.
        """
        text = self.conn.drawing.add_text(text=label, reference_frame=self.text_refframe, position=label_pos, rotation=rot)
        text.color = color
        text.anchor = self.conn.ui.TextAnchor.middle_left
        return text

    def _normalize_vectors(self, vectors):
        """
        Normalizes vectors to unit length.
        :param vectors: A single vector (1D) or multiple vectors (2D array).
        :return: Normalized vectors.
        """
        return np.array([vec / np.linalg.norm(vec) for vec in vectors])

    def _get_random_color(self, dim):
        """Returns a random color from the XKCD color list."""
        colors = []
        for i in range(0, dim):
            hex_color = random.choice(self.xkcd_colors)  # Pick a random hex color
            colors.append(mcolors.to_rgb(hex_color))  # Convert hex to an RGB tuple
        return colors

    def clear_vectors(self):
        """Clears all currently drawn vectors."""
        self.conn.drawing.clear()
        for vector_data in self.vector_funcs.values():
            vector_data["line_objs"] = None
            vector_data["text_objs"] = None

    def start_drawing(self, dt=0.1):
        """Starts automatic updating of vectors in a separate thread with interval dt."""
        if not self._updating:
            self._updating = True
            self._update_thread = threading.Thread(target=self._auto_update, args=(dt,))
            self._update_thread.start()
            print("Automatic vector updating started.")

    def _auto_update(self, dt):
        """Runs draw_vectors at specified intervals until stopped."""
        while self._updating:
            self.draw_vectors()
            time.sleep(dt)

    def stop_drawing(self):
        """Stops the automatic updating thread and clears vectors."""
        if self._updating:
            self._updating = False
            self._update_thread.join()
            self.clear_vectors()  # Clear vectors on stop
            print("Automatic vector updating stopped and vectors cleared.")

    def _compute_text_rotation(self, vector, fixed_orientation):
        
        # Reference direction in local frame
        reference = np.array([1, 0, 0])  # Change as needed
        
        # Compute alignment quaternion
        cross = np.cross(reference, vector)
        dot = np.dot(reference, vector)
        align_quat = np.array([*cross, 1 + dot])
        align_quat /= np.linalg.norm(align_quat)  # Normalize quaternion
        
        # Fixed orientation quaternion
        fixed_quat = R.from_euler('xyz', fixed_orientation).as_quat()
        
        # Combine quaternions
        final_quat = R.from_quat(align_quat) * R.from_quat(fixed_quat)
        return final_quat.as_quat()
