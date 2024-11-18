# Robot Simulation

## Overview

Robot Simulation is a Python-based application designed to simulate the movement of a robotic manipulator. Given a set of X, Y, and Z coordinates, the program generates a video representation of the robot's movements, showcasing its inverse kinematics capabilities. This project leverages Mujoco for physics simulation, IKPy for inverse kinematics calculations, and several Python libraries for visualization and file handling.

## Features

- **Simulation Modes**:  
  - `headless-simulation.py`: Runs the simulation without graphical rendering for faster performance.  
  - `simulation.py`: Provides a graphical interface for visualizing the simulation.

- **Inverse Kinematics**: Calculate and control the robot's movement using IKPy.  
- **Video Generation**: Save simulation outputs as videos using `imageio`.  
- **Customizable Environment**: Use XML-based scene descriptions to configure the simulation environment.  

## Requirements

The project relies on the following Python packages:

```plaintext
absl-py==2.1.0
certifi==2024.8.30
... (Refer to the provided `requirements.txt` file for the complete list.)
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

Here is the updated installation section with instructions for creating a virtual environment on both macOS and Windows:

## Installation

1. **Clone the Repository**:  

   ```bash
   git clone https://github.com/your-username/robot-simulation.git
   cd robot-simulation
   ```

2. **Set Up a Virtual Environment**:

   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   This ensures all dependencies are isolated within the project environment.

3. **Install the Required Dependencies**:  

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Mujoco**:  
   Ensure you have Mujoco installed and properly configured. Refer to the [Mujoco Documentation](https://mujoco.org/) for platform-specific setup instructions.

5. **Run a Sample Simulation to Validate**:  

   ```bash
   python simulation.py
   ```

## Usage

### Running Simulations

- **Graphical Mode**:  
  Run the simulation with a user interface:  
  ```bash
  python simulation.py
  ```

- **Headless Mode**:  
  For performance optimization and background execution:  
  ```bash
  python headless-simulation.py
  ```

### Customizing the Environment

- **Scene Configuration**: Modify the `scene.xml` or `scene_with_obj.xml` files to customize the simulation environment.  
- **Robot Definition**: Update the `TerafacMini.xml` file to configure the robot's joints, geometry, and dynamics.  

## Acknowledgements

This project uses the following technologies:

- [Mujoco](https://mujoco.org/) for dynamic simulation.  
- [IKPy](https://github.com/Phylliade/ikpy) for inverse kinematics.  
- Python libraries such as NumPy, Matplotlib, and ImageIO for computational and visualization support.
