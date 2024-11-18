import mujoco
import numpy as np
import time
import json
import os
import ikpy.chain
import logging
import requests
from urllib.parse import urlparse
import queue
from datetime import datetime
import imageio
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OBJ_URL = "https://terafac-welding-pro.s3.amazonaws.com/welding_objects/Corner_Joint_fe2890e3.obj"

WORKBENCH_SIZE = "0.1240 0.1767 0.001"
WORKBENCH_POSITION = "0.3 0 0.001"
MESH_SCALE = "0.002 0.002 0.002"
MESH_POSITION = "0.3 0 0.061"


class MujocoSimulator:
    def __init__(self):
        self.model = None
        self.data = None
        self.my_chain = None
        self.running = False
        self.command_queue = queue.Queue()
        self.movement_speed = 0.01
        self.ee_site_id = None
        self.current_pos = None
        self.paused = False
        self.movement_history = []
        self.frames = []
        self.recording = False
        self.video_dir = "simulation_videos"
        self.last_simulation = None
        self.renderer = None

        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def start_recording(self):
        """Start capturing frames for video recording"""
        self.frames = []
        self.recording = True

        self.renderer = mujoco.Renderer(self.model, width=640, height=480)

    def stop_recording(self):
        """Stop recording and save the video"""
        self.recording = False
        if self.frames:
            self.save_video()

        self.renderer = None

    def save_video(self):
        """Save recorded frames as an MP4 video with memory-efficient encoding"""
        if not self.frames:
            logger.warning("No frames to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"SIM-{timestamp}.mp4"
        filepath = os.path.join(self.video_dir, filename)

        try:

            frames_uint8 = [
                (
                    (frame * 255).astype(np.uint8)
                    if frame.dtype == np.float32
                    else frame.astype(np.uint8)
                )
                for frame in self.frames
            ]

            writer = imageio.get_writer(
                filepath,
                fps=30,
                codec="h264",
                quality=7,
                macro_block_size=16,
                bitrate=2000000,
                output_params=[
                    "-preset",
                    "medium",
                    "-tune",
                    "zerolatency",
                    "-profile:v",
                    "baseline",
                    "-pix_fmt",
                    "yuv420p",
                ],
            )

            chunk_size = 30
            for i in range(0, len(frames_uint8), chunk_size):
                chunk = frames_uint8[i : i + chunk_size]
                for frame in chunk:
                    writer.append_data(frame)
                logger.info(
                    f"Processed frames {i} to {min(i + chunk_size, len(frames_uint8))}"
                )

            writer.close()
            self.last_simulation = filepath
            logger.info(f"Video saved successfully: {filepath}")

        except MemoryError:
            logger.error(
                "Memory error while saving video. Attempting alternative saving method..."
            )
            try:

                imageio.mimsave(
                    filepath,
                    frames_uint8,
                    fps=15,
                    quality=5,
                    macro_block_size=16,
                    output_params=["-preset", "ultrafast"],
                )
                self.last_simulation = filepath
                logger.info(
                    f"Video saved successfully with alternative method: {filepath}"
                )
            except Exception as e:
                logger.error(f"Alternative saving method failed: {e}")

                frames_dir = os.path.join(self.video_dir, f"frames_{timestamp}")
                os.makedirs(frames_dir, exist_ok=True)
                for i, frame in enumerate(frames_uint8):
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    imageio.imwrite(frame_path, frame)
                logger.info(f"Saved individual frames to: {frames_dir}")
                self.last_simulation = frames_dir

        except Exception as e:
            logger.error(f"Error saving video: {e}")
            if hasattr(e, "output"):
                logger.error(f"Error output: {e.output}")
        finally:
            self.frames = []

            import gc

            gc.collect()

    def capture_frame(self):
        """Capture the current frame using offscreen rendering with memory management"""
        if self.recording and self.renderer is not None:
            try:
                self.renderer.update_scene(self.data, "tracker")
                pixels = self.renderer.render()
                if pixels is not None:

                    if pixels.dtype == np.float32:
                        pixels = (pixels * 255).astype(np.uint8)

                    self.frames.append(pixels)

                    if len(self.frames) > 1000:
                        self.save_video()
                        self.frames = []

            except Exception as e:
                logger.error(f"Error capturing frame: {e}")

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

    def create_scene_with_obj_and_marker(
        self, obj_filepath, marker_filepath, scene_template="scene.xml"
    ):
        """Creates a new scene XML file that includes both the OBJ file and Aruco marker"""
        tree = ET.parse(scene_template)
        root = tree.getroot()

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
        geom.set("rgba", "0.8 0.8 0.8 1")

        new_scene_path = "scene_with_obj.xml"
        tree.write(new_scene_path)
        return new_scene_path

    def load_obj_and_setup(self, obj_url, marker_url=None):
        """Loads OBJ file and sets up the simulation without viewer"""
        try:
            obj_filepath = self.download_obj_file(obj_url)

            marker_filepath = None
            if marker_url or marker_url == "":
                marker_filepath = self.download_image(marker_url)

            if marker_filepath:
                scene_path = self.create_scene_with_obj_and_marker(
                    obj_filepath, marker_filepath
                )
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
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
            )

            logger.info("Successfully loaded OBJ file and set up simulation")

        except Exception as e:
            logger.error(f"Error loading OBJ file: {e}")
            raise

    def move_to_position(self, x, y, z):
        try:
            self.stop_current_movement = False
            x *= -1
            y *= -1
            target_position = [x, y, z]
            joint_angles = self.my_chain.inverse_kinematics(target_position)
            achieved_position = self.my_chain.forward_kinematics(joint_angles)[:3, 3]

            logger.info(f"Moving to target position: [{x:.2f}, {y:.2f}, {z:.2f}]")
            logger.info(
                f"Achieved position: [{achieved_position[0]:.2f}, {achieved_position[1]:.2f}, {achieved_position[2]:.2f}]"
            )

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

                current_pos = self.data.site_xpos[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
                    )
                ].copy()
                self.current_movement_path.append(current_pos)

                self.capture_frame()

                progress = (i + 1) / num_steps * 100
                logger.info(f"Movement progress: {progress:.1f}%")

                time.sleep(self.movement_speed)

            if not self.stop_current_movement:
                self.movement_history.append(self.current_movement_path)
                self.data.qpos[:] = safe_pos
                self.current_pos = safe_pos
                mujoco.mj_step(self.model, self.data)

            logger.info("Movement completed!")
            final_pos = self.data.site_xpos[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
                )
            ]
            logger.info(
                f"Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]"
            )

        except Exception as e:
            logger.error(f"Error during movement: {str(e)}")

    def load_scene(self, data):
        obj_url = data.get("obj_url", DEFAULT_OBJ_URL)
        marker_url = data.get("marker_url", "")
        self.load_obj_and_setup(obj_url, marker_url)

    def simulate(self, x, y, z):
        self.command_queue.put((x, y, z))


def run_simulation(coordinates, obj_url, simulator):
    data = {"obj_url": obj_url if obj_url else DEFAULT_OBJ_URL, "marker_url": ""}
    simulator.load_scene(data)

    simulator.running = True
    recording_started = False

    for x, y, z in coordinates:
        simulator.command_queue.put((x, y, z))

    while simulator.running:
        if not simulator.command_queue.empty():
            if not recording_started:
                simulator.start_recording()
                recording_started = True

            cmd = simulator.command_queue.get_nowait()
            simulator.move_to_position(*cmd)

        if simulator.command_queue.empty() and recording_started:
            simulator.stop_recording()
            break

        time.sleep(0.1)

    return simulator.last_simulation


if __name__ == "__main__":
    simulator = MujocoSimulator()
    coordinates = [(0.1, 0.1, 0.1), (0.1, 0.2, 0.1), (0.2, 0.2, 0.1), (0.2, 0.2, 0.2)]
    run_simulation(coordinates, "", simulator)
