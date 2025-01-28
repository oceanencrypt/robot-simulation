import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import os
import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import xml.etree.ElementTree as ET
import logging
import requests
from urllib.parse import urlparse
import tkinter as tk
from tkinter import ttk
import threading
import queue
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glfw
from datetime import datetime
import cv2
from mujoco import gl_context
import imageio
import glfw
from coordinates import sio
import sys
import json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

welding_data = None

CONTROL_PANEL = False

DEFAULT_OBJ_URL = "https://terafac-pnc.s3.amazonaws.com/welding_objects/cuboid.obj"

WORKBENCH_SIZE = "0.1240 0.1767 0.001" #0.1240 0.1767 0.001
WORKBENCH_POSITION = "0.3 0 0.001" # The centroid of the workbench.

MESH_SCALE = "0.0008 0.0008 0.0008" #MESH_SCALE = "0.002 0.002 0.002"0.0008 0.0008 0.0008
MESH_POSITION = "0.3 0 0.001" #0.04037 -0.1279 -0.1715
#0.3 0 0.061



def calculate_bounding_box(obj_file_path):
    """
    Calculate the bounding box of an .obj file.
    Returns the center, dimensions, and minimum z-coordinate of the bounding box.
    """
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    with open(obj_file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex data
                _, x, y, z = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
                max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)

    # Center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    # Dimensions of the bounding box
    dimensions = (max_x - min_x, max_y - min_y, max_z - min_z)

    return (center_x, center_y, center_z), dimensions, min_z


def apply_scale(center, dimensions, mesh_scale):
    """
    Scale the bounding box center and dimensions.
    """
    scale_x, scale_y, scale_z = map(float, mesh_scale.split())
    scaled_center = (center[0] * scale_x, center[1] * scale_y, center[2] * scale_z)
    scaled_dimensions = (dimensions[0] * scale_x, dimensions[1] * scale_y, dimensions[2] * scale_z)
    return scaled_center, scaled_dimensions

def calculate_mesh_position(obj_center, workbench_position, mesh_scale, min_z):
    """
    Align the object's center to the workbench center and adjust the z-coordinate to sit on the workbench.
    """
    workbench_x, workbench_y, workbench_z = map(float, workbench_position.split())
    scale_x, scale_y, scale_z = map(float, mesh_scale.split())

    # Offset for x and y (center alignment)
    offset_x = workbench_x - obj_center[0]
    offset_y = workbench_y - obj_center[1]

    # Adjust z-coordinate to align the bottom of the object with the workbench
    offset_z = workbench_z - (min_z * scale_z)
    print(f"z offset: {offset_z}")
    print(f"min z : {min_z}")
    return f"{offset_x:.5f} {offset_y:.5f} {offset_z:.5f}"



class RobotControlGUI:
    def __init__(self, simulator):
        self.simulator = simulator
        self.root = tk.Tk()
        self.root.title("Advanced Robot Control Panel")

        self.x_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=0.0)
        self.speed_var = tk.DoubleVar(value=2000) # value = 0.01
        self.joint_labels = []

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.create_widgets()
        self.setup_path_visualization()

        self.sim_thread = threading.Thread(target=self.simulator.run_simulation_GUI)
        self.sim_thread.daemon = True
        self.sim_thread.start()

        self.update_joint_angles()
        self.update_path_visualization()

    def create_widgets(self):

        control_frame = ttk.LabelFrame(self.root, text="Position Control", padding="10")
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        positions = [("X:", self.x_var), ("Y:", self.y_var), ("Z:", self.z_var)]
        for i, (label, var) in enumerate(positions):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(control_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Speed:").grid(row=3, column=0, padx=5, pady=5)
        self.speed_scale = ttk.Scale(
            control_frame,
            from_=0.0001,
            to=0.05,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
        )
        self.speed_scale.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Move", command=self.move_robot).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Pause/Resume", command=self.toggle_pause).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Stop", command=self.stop_movement).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Clear Path", command=self.clear_path).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Weld", command=self.weld).pack(
            side=tk.LEFT, padx=5
        )

        joint_frame = ttk.LabelFrame(self.root, text="Joint Angles", padding="10")
        joint_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.joint_labels = []
        for i in range(6):
            label = ttk.Label(joint_frame, text=f"Joint {i+1}: 0.000 rad")
            label.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.joint_labels.append(label)


    def setup_path_visualization(self):

        path_frame = ttk.LabelFrame(self.root, text="Path Visualization", padding="10")
        path_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=path_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Robot Path")

    def update_path_visualization(self):
        if self.simulator.current_movement_path:
            self.ax.cla()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            for path in self.simulator.movement_history[:-1]:
                if path:
                    path = np.array(path)
                    self.ax.plot(path[:, 0], path[:, 1], path[:, 2], "gray", alpha=0.3)

            if self.simulator.movement_history:
                current_path = np.array(self.simulator.current_movement_path)
                self.ax.plot(
                    current_path[:, 0], current_path[:, 1], current_path[:, 2], "b-"
                )

            self.canvas.draw()

        self.root.after(100, self.update_path_visualization)

    def update_joint_angles(self):
        angles = self.simulator.get_joint_angles()
        for i, angle in enumerate(angles):
            self.joint_labels[i].config(text=f"Joint {i+1}: {angle:.3f} rad")
        self.root.after(100, self.update_joint_angles)

    def move_robot(self):
        x = self.x_var.get()
        y = self.y_var.get()
        z = self.z_var.get()
        self.simulator.movement_speed = self.speed_var.get()
        self.simulator.command_queue.put((y, x, z))

    def weld(self):
        self.simulator.weld()

    def toggle_pause(self):
        self.simulator.paused = not self.simulator.paused

    def stop_movement(self):
        self.simulator.stop_current_movement = True
        self.simulator.paused = False

    def clear_path(self):
        self.simulator.movement_history = []
        self.simulator.current_movement_path = []

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.simulator.running = False
        self.root.destroy()


class MujocoSimulator:
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        self.renderer = None
        self.my_chain = None
        self.joint_limits = None
        self.running = False
        self.command_queue = queue.Queue()
        self.movement_speed = 0.01
        self.ee_site_id = None
        self.current_pos = None
        # self.load_scene({})
        self.paused = False
        self.movement_history = []
        self.frames = []
        self.recording = False
        self.video_dir = "simulation_videos"
        self.last_simulation = None
        self.current_movement_path = []
        self.last_update_time = 0

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def start_recording(self):
        """Start capturing frames for video recording"""
        self.frames = []
        self.recording = True

    def stop_recording(self):
        """Stop recording and save the video"""
        self.recording = False
        if self.frames:
            self.save_video()

    def capture_frame(self):
        """Capture the current viewer frame"""
        if self.recording and self.viewer is not None:
            try:
                self.renderer.update_scene(self.data, "tracker")
                pixels = self.renderer.render()
                if pixels is not None:
                    self.frames.append(pixels)
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")

    def save_video(self):
        """Save recorded frames as an MP4 video"""
        if not self.frames:
            logger.warning("No frames to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"SIM-{timestamp}.mp4"
        filepath = os.path.join(self.video_dir, filename)

        try:

            imageio.mimsave(
                filepath,
                self.frames,
                fps=60,
                quality=9,
                macro_block_size=0,
            )
            self.last_simulation = filepath
            logger.info(f"Video saved successfully: {filepath}")

        except Exception as e:
            logger.error(f"Error saving video: {e}")
        finally:
            self.frames = []

    def download_obj_file(self, url, save_dir="assets"):
        """Downloads an OBJ file from URL and saves it locally"""
        try:

            if not os.path.exists(save_dir):
                logger.info(f"Creating directory: {save_dir}")
                os.makedirs(save_dir)

            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            filepath = os.path.join(save_dir, filename)

            logger.info(f"Target filepath: {filepath}")

            if not os.path.exists(filepath):
                logger.info(f"Downloading OBJ file from {url}")
                response = requests.get(url)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                logger.info(f"Downloaded OBJ file to {filepath}")
                logger.info(f"File size: {os.path.getsize(filepath)} bytes")
            else:
                logger.info(f"File already exists at {filepath}")
                logger.info(f"Existing file size: {os.path.getsize(filepath)} bytes")

            return filepath

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download OBJ file: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to save OBJ file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while downloading OBJ file: {e}")
            raise

    def download_image(self, url, save_dir="assets"):
        """Downloads an image from URL and saves it locally"""
        try:
            if not os.path.exists(save_dir):
                logger.info(f"Creating directory: {save_dir}")
                os.makedirs(save_dir)

            filename = "workbench.png"

            if url:
                parsed_url = urlparse(url)
                filename = f"marker_{os.path.basename(parsed_url.path)}"

            if not any(
                filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]
            ):
                filename += ".png"

            filepath = os.path.join(save_dir, filename)

            logger.info(f"Target filepath: {filepath}")

            if not os.path.exists(filepath):
                logger.info(f"Downloading image from {url}")
                response = requests.get(url)
                response.raise_for_status()

                from PIL import Image
                import io

                image = Image.open(io.BytesIO(response.content))

                if image.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                image.save(filepath, "PNG")

                logger.info(f"Downloaded image to {filepath}")
                logger.info(f"File size: {os.path.getsize(filepath)} bytes")
            else:
                logger.info(f"File already exists at {filepath}")
                logger.info(f"Existing file size: {os.path.getsize(filepath)} bytes")

            return filepath

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to save image: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while downloading image: {e}")
            raise

    def load_obj_and_setup(self, obj_url, marker_url=None):
        """
        Dynamically loads an OBJ file and updates its position to align with the workbench center
        and sit properly on the workbench surface.
        """
        try:
            # Download OBJ file
            obj_filepath = self.download_obj_file(obj_url)

            if os.path.exists(obj_filepath):
                logger.info("✓ File download successful")
                logger.info(f"✓ File saved at: {obj_filepath}")
                logger.info(f"✓ File size: {os.path.getsize(obj_filepath)} bytes")

            # Calculate bounding box and scaled center
            obj_center, obj_dimensions, min_z = calculate_bounding_box(obj_filepath)
            print(f"The center of the obj is {obj_center}")
            scaled_center, _ = apply_scale(obj_center, obj_dimensions, MESH_SCALE)
            print(f"The scaled center of the obj is {scaled_center}")

            # Update MESH_POSITION dynamically
            global MESH_POSITION
            MESH_POSITION = calculate_mesh_position(scaled_center, WORKBENCH_POSITION, MESH_SCALE, min_z)
            logger.info(f"Dynamically calculated MESH_POSITION: {MESH_POSITION}")

            # Proceed with loading the object into the scene
            if marker_url or marker_url == "":
                marker_filepath = self.download_image(marker_url)
                scene_path = self.create_scene_with_obj_and_marker(obj_filepath, marker_filepath)
            else:
                scene_path = self.create_scene_with_obj(obj_filepath)

            self.model = mujoco.MjModel.from_xml_path(scene_path)
            self.data = mujoco.MjData(self.model)

            self.ee_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )

            key_name = "home"
            key_id = self.model.key(key_name).id
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

            self.my_chain = ikpy.chain.Chain.from_urdf_file(
                "Tera.urdf",
                active_links_mask=[
                    False, True, False, True, False, True,
                    False, True, False, True, False, True,
                    False, False, False,
                ],
            )

            if self.viewer:
                self.viewer.close()

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)

            logger.info("Successfully loaded OBJ file and set up simulation")

        except Exception as e:
            logger.error(f"Error loading OBJ file: {e}")
            raise



#     def create_scene_with_obj_and_marker(
#     self, obj_filepath, marker_filepath, scene_template="scene.xml"
# ):
#         """
#         Creates a new scene XML file that includes both the OBJ file and Aruco marker.
#         """
#         tree = ET.parse(scene_template)
#         root = tree.getroot()

#         asset = root.find("asset")
#         if asset is None:
#             asset = ET.SubElement(root, "asset")

#         mesh = ET.SubElement(asset, "mesh")
#         mesh.set("name", "welding_mesh")
#         mesh.set("file", obj_filepath)
#         mesh.set("scale", MESH_SCALE)

#         texture = ET.SubElement(asset, "texture")
#         texture.set("name", "marker_texture")
#         texture.set("type", "2d")
#         texture.set("file", marker_filepath)
#         texture.set("width", "512")
#         texture.set("height", "731")

#         material = ET.SubElement(asset, "material")
#         material.set("name", "marker_material")
#         material.set("texture", "marker_texture")
#         material.set("texrepeat", "1 1")

#         worldbody = root.find("worldbody")

#         marker_body = ET.SubElement(worldbody, "body")
#         marker_body.set("name", "marker_plane")
#         marker_body.set("pos", WORKBENCH_POSITION)

#         marker_geom = ET.SubElement(marker_body, "geom")
#         marker_geom.set("type", "plane")
#         marker_geom.set("size", WORKBENCH_SIZE)
#         marker_geom.set("material", "marker_material")

#         body = ET.SubElement(worldbody, "body")
#         body.set("name", "welding_object")
#         body.set("pos", MESH_POSITION)

#         geom = ET.SubElement(body, "geom")
#         geom.set("type", "mesh")
#         geom.set("mesh", "welding_mesh")
#         geom.set("rgba", "0.8 0.8 0.8 1")
        

#         new_scene_path = "scene_with_obj.xml"
#         tree.write(new_scene_path)
#         return new_scene_path

    def create_scene_with_obj_and_marker(
        self, obj_filepath, marker_filepath, scene_template="scene.xml"
    ):
        """Creates a new scene XML file that includes both the OBJ file and Aruco marker with collision detection"""
        tree = ET.parse(scene_template)
        root = tree.getroot()

        contact = ET.Element("contact")
        pair = ET.SubElement(contact, "pair")
        pair.set("geom1", "link6_collision")
        pair.set("geom2", "welding_mesh_collision")
        pair.set("margin", "0.0001")  # 0.1mm safety margin
        pair.set("solref", "0.01 1")  # Quicker contact response
        pair.set("solimp", "0.95 0.99 0.001")
        root.insert(1, contact)

        default = root.find("default")
        if default is None:
            default = ET.Element("default")
            root.insert(1, default)

        col_default = ET.SubElement(default, "default")
        col_default.set("class", "collision")
        col_geom = ET.SubElement(col_default, "geom")
        col_geom.set("contype", "1")
        col_geom.set("conaffinity", "1")
        col_geom.set("condim", "3")
        col_geom.set("priority", "1")

        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")

        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")

        flag.set("contact", "enable")
        flag.set("filterparent", "disable")
        option.set("timestep", "0.002")

        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        mesh = ET.SubElement(asset, "mesh")
        mesh.set("name", "welding_mesh")
        mesh.set("file", obj_filepath)
        mesh.set("scale", MESH_SCALE)

        texture = ET.SubElement(asset, "texture")
        texture.set("name", "marker_texture")
        texture.set("type", "2d")
        texture.set("file", marker_filepath)
        texture.set("width", "512")
        texture.set("height", "731")

        material = ET.SubElement(asset, "material")
        material.set("name", "marker_material")
        material.set("texture", "marker_texture")
        material.set("texrepeat", "1 1")

        worldbody = root.find("worldbody")

        marker_body = ET.SubElement(worldbody, "body")
        marker_body.set("name", "marker_plane")
        marker_body.set("pos", WORKBENCH_POSITION)

        marker_geom = ET.SubElement(marker_body, "geom")
        marker_geom.set("type", "plane")
        marker_geom.set("size", WORKBENCH_SIZE)
        marker_geom.set("material", "marker_material")

        body = ET.SubElement(worldbody, "body")
        body.set("name", "welding_object")
        body.set("pos", MESH_POSITION)

        geom = ET.SubElement(body, "geom")
        geom.set("type", "mesh")
        geom.set("mesh", "welding_mesh")
        geom.set("group", "1")

        collision = ET.SubElement(body, "geom")
        collision.set("name", "welding_mesh_collision")
        collision.set("type", "mesh")
        collision.set("mesh", "welding_mesh")
        collision.set("class", "collision")
        collision.set("rgba", "1 0 0 0.3")
        collision.set("margin", "0.0001") 

        new_scene_path = "scene_with_obj.xml"
        tree.write(new_scene_path)

        robot_tree = ET.parse("TerafacMini.xml")
        robot_root = robot_tree.getroot()

        link6 = robot_root.find(".//body[@name='link6']")
        if link6 is not None:
            collision = ET.SubElement(link6, "geom")
            collision.set("name", "link6_collision")
            collision.set("type", "mesh")
            collision.set("mesh", "link6")
            collision.set("rgba", "1 0 0 0.3")
            collision.set("class", "collision")
            collision.set("margin", "0.005")

        robot_tree.write("TerafacMini_with_collision.xml")

        tree = ET.parse(new_scene_path)
        root = tree.getroot()
        include = root.find("include")
        include.set("file", "TerafacMini_with_collision.xml")
        tree.write(new_scene_path)

        return new_scene_path


    def check_collision(self):
        """
        Check for collisions with improved detection and reporting
        """

        link6_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "link6_collision"
        )
        mesh_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "welding_mesh_collision"
        )

        force = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (contact.geom1 == link6_id and contact.geom2 == mesh_id) or (
                contact.geom1 == mesh_id and contact.geom2 == link6_id
            ):

                mujoco.mj_contactForce(self.model, self.data, i, force)
                force_magnitude = np.linalg.norm(force[:3])

                if contact.dist < -0.000001:
                    print(
                        f"Collision detected! Distance: {contact.dist}, Force: {force_magnitude:.3f}"
                    )
                    return True

        return False


    def move_to_position(self, x, y, z):
        try:
            self.stop_current_movement = False
            x *= -1
            y *= -1
            target_position = [x, y, z]
            joint_angles = self.my_chain.inverse_kinematics(target_position)
            achieved_position = self.my_chain.forward_kinematics(joint_angles)[:3, 3]

            print(f"Moving to target position: [{x:.2f}, {y:.2f}, {z:.2f}]")
            print(
                f"Achieved position: [{achieved_position[0]:.2f}, {achieved_position[1]:.2f}, {achieved_position[2]:.2f}]"
            )

            print("CURRENT _POS :::", self.current_pos)
            current_qpos = (
                self.current_pos
                if hasattr(self, "current_pos")
                and self.current_pos is not None
                and np.any(self.current_pos)
                else self.data.qpos.copy()
            )
            new_qpos = current_qpos.copy()
            matrix = np.array(joint_angles)
            odd_indices = np.arange(1, 12, 2)
            non_zero_values_at_odd_indices = matrix[odd_indices]
            new_qpos[0:6] = non_zero_values_at_odd_indices

            print("CURRENT POS :::", current_qpos)
            print("FINAL POS :::", new_qpos)

            num_steps = 140
            self.current_movement_path = []

            def cubic_ease(t):
                return t * t * (3 - 2 * t)

            for i in range(num_steps + 1):
                if self.stop_current_movement:
                    break

                while self.paused and not self.stop_current_movement:
                    time.sleep(0.1)

                t = i / num_steps
                t = cubic_ease(t)
                pos = current_qpos + (new_qpos - current_qpos) * t

                safe_pos = pos
                self.data.qpos[:] = safe_pos

                mujoco.mj_step(self.model, self.data)

                #self.check_collision()
                # Check for collision after movement
                if self.check_collision():
                    print("Collision occurred during movement!")
                else:
                    print("No collision detected after movement.")

                current_pos = self.data.site_xpos[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
                    )
                ].copy()
                self.current_movement_path.append(current_pos)

                if self.viewer is not None:
                    self.viewer.sync()
                    self.capture_frame()

                progress = (i + 1) / len(range(num_steps + 1)) * 100
                print(f"\rMovement progress: {progress:.1f}%", end="", flush=True)
                #print("gfjdsg: ", self.my_chain)
                # Call the collision detection function
        
                time.sleep(0.00002) #0.002

            if not self.stop_current_movement:
                self.movement_history.append(self.current_movement_path)

                self.data.qpos[:] = safe_pos
                self.current_pos = safe_pos
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self.viewer.sync()

            print("\nMovement completed!")
            final_pos = self.data.site_xpos[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
                )
            ]
            print(
                f"Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]"
            )
            
               
            

        except Exception as e:
            print(f"Error during movement: {str(e)}")





    def run_simulation(self):
        # Load the scene with the given object URL
        
        self.running = True
        while self.running:
            if self.viewer is not None and not CONTROL_PANEL:
                try:
                    user_input = input("Enter command (simulate/move/quit): ")
                    if user_input.lower() == "quit":
                        self.viewer.close()
                        break
                    elif user_input.lower() == "simulate":
                        simulator.simulate(0.1, 0.1, 0.1)
                    elif user_input.lower() == "move":
                        target_input = input("Enter new target position (x y z): ")
                        x, y, z = map(float, target_input.split())
                        simulator.simulate(y, x, z)
                    cmd = self.command_queue.get_nowait()
                    self.move_to_position(*cmd)
                except ValueError as e:
                    print(f"Invalid input: {e}")

                self.viewer.sync()
                if not self.viewer.is_running():
                    break

            time.sleep(0.0001)

        self.running = False
        if self.viewer is not None:
            self.viewer.close()

    def run_simulation_GUI(self):
        self.running = True
        while self.running:
            if self.viewer is not None:
                current_time = time.time()
                if current_time - self.last_update_time >= 0.001:
                    self.last_update_time = current_time

                
                
                self.viewer.sync()
                
                try:
                    if self.command_queue.qsize() <= 1:
                        self.done_welding = True
                    else:
                        self.done_welding = False
                    cmd = self.command_queue.get_nowait()
                    self.move_to_position(*cmd)
                except queue.Empty:
                    pass
                
                if not self.viewer.is_running():
                    break
                    
            time.sleep(0.001)
        
        self.running = False
        if self.viewer is not None:
            self.viewer.close()
  
    def get_joint_angles(self):
        if self.data is not None:
            return self.data.qpos[:6].copy()
        return np.zeros(6)

    def load_scene(self, data):
        obj_url = data["obj_url"] if data.get("obj_url", None) else DEFAULT_OBJ_URL
        marker_url = data["marker_url"] if data.get("marker_url", None) else ""
        self.load_obj_and_setup(obj_url, marker_url)

    def simulate(self, x, y, z):
        self.command_queue.put((y, x, z))


def run_simulation(coordinates, obj_url, simulator):

    data = {"obj_url": obj_url if obj_url else DEFAULT_OBJ_URL, "marker_url": ""}
    simulator.load_scene(data)

    simulator.running = True

    recording_started = False

    for x, y, z in coordinates:
        simulator.command_queue.put((y, x, z))

    while simulator.running:
        if not simulator.command_queue.empty():

            cmd = simulator.command_queue.get_nowait()

            if not recording_started:
                simulator.start_recording()
                recording_started = True

            simulator.move_to_position(*cmd)

        if simulator.command_queue.empty() and recording_started:
            simulator.stop_recording()
            break

        time.sleep(0.1)

    simulator.running = True    
    simulator.viewer.close()

    return simulator.last_simulation


if __name__ == "__main__":
    simulator = MujocoSimulator()
    
    
    if CONTROL_PANEL:
        simulator.load_scene({})
        gui = RobotControlGUI(simulator)
        gui.run()
        
    else:
        # print("Waiting for welding data...")
        # while welding_data is None:
        #     time.sleep(1)  # Wait for data from the SocketIO server

        # coordinates = welding_data['coordinates']
        # obj_url = welding_data['obj_url']

        # print(f"Running simulation with coordinates: {coordinates}")
        # print(f"Using OBJ URL: {obj_url}")

        # run_simulation(coordinates, obj_url, simulator)
        if __name__ == "__main__":
            simulator = MujocoSimulator()

            if len(sys.argv) > 1:
                # Parse the JSON string from the command-line argument
                try:
                    welding_data = json.loads(sys.argv[1])
                    coordinates = welding_data['coordinates']
                    obj_url = welding_data['obj_url']

                    print(f"Running simulation with coordinates: {coordinates}")
                    print(f"Using OBJ URL: {obj_url}")

                    run_simulation(coordinates, obj_url, simulator)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Invalid welding data provided: {e}")
                    sys.exit(1)
            else:
                print("No welding data provided. Exiting.")
                sys.exit(1)

