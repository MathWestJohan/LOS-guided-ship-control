# LOS-guided-ship-control

Project repository for course MMA4007 Applied AI and Control.

A ship autopilot simulation using Line-Of-Sight (LOS) guidance with PID control in the [AGX Dynamics](https://www.algoryx.se/agx-dynamics/) physics engine. The ship follows a randomly generated route of waypoints to a fixed dock point using classic marine guidance and control theory. The LOS-PID controller is used as an expert demonstrator to collect training data, which is then used to train a pair of MLP networks that learn to imitate the controller — the long-term goal is an ML-assisted autonomous docking system.

## How it works

### Guidance and control pipeline

LOS guidance computes the desired heading from the ship's cross-track error relative to the path between waypoints. Reference filters smooth the heading and speed commands. A PID controller produces surge, sway, and yaw forces/moments. Thrust allocation distributes the commands to two stern azimuth thrusters. A state observer filters noisy GNSS measurements and estimates velocities.

### Random route generation

Each run, a route is generated from a randomized start position and heading inside a configurable initial-area zone, terminating at the fixed dock point. This produces a diverse set of approach trajectories for training.

### Data collection and learning

The simulation logs all states, references, and control signals to CSV. A headless AGX runner ([data_collection.py](src/data_collection.py)) batches many episodes without the visual viewer for fast dataset generation. Two MLPs are then trained on the logged demonstrations:

- **Speed network:** `[dx, dy, u, u_r, e_ct, dist_dock] → τ_x`
- **Steering network:** `[sin(ψ), cos(ψ), e_ct, e_ψ, v, r, dx, dy] → τ_y, τ_ψ`

At inference time, [model_controller.py](src/model/model_controller.py) drops in as a replacement for the PID controller inside the runner.

## Project layout

```
src/
  main.py                  # AGX visual entry point — launches runner with viewer
  data_collection.py       # Headless multi-episode dataset generator
  plot_log.py              # Plot trajectories and signals from logged CSV
  agx_wrap/                # AGX scene helpers (ocean, world)
  modeling/                # Vessel rigid-body model
  control/                 # LOS guidance, reference filter, PID, observer,
                           # thrust allocation, random route generator
  runtime/                 # Scene builder, step callback, config
  model/                   # MLP training pipeline and inference controller
data/                      # Logged CSVs and plots (gitignored data dumps)
assets/                    # Ship mesh (Gunnerus.obj)
```

## Requirements

- AGX Dynamics with Python bindings (tested with 2.40.1.5)
- Python 3.9.9
- NumPy, Matplotlib
- For training only: PyTorch, pandas, scikit-learn

## Running

### Visual simulation (single run with viewer):

```bat
"%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5\python-x64\python.exe" src/main.py
```

### Headless data collection (many episodes, no viewer):

```bat
"%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5\python-x64\python.exe" src/data_collection.py --episodes 50 --seed 0 --output data/training_data.csv
```

### Train the MLP controllers:

```bat
python src/model/train_mlp.py
```

### One-time AGX environment setup

If `import agx` fails outside the AGX-bundled Python, configure the environment:

```bat
.\.venv\Scripts\activate
set "AGX_DIR=%LOCALAPPDATA%\Algoryx\AGX-2.40.1.5"
set "PATH=%AGX_DIR%\bin\x64;%PATH%"
set "PYTHONPATH=%AGX_DIR%\bin\x64\agxpy;%AGX_DIR%\data\python\modules;%AGX_DIR%\data\python"
python -c "import agx, agxSDK, agxPythonModules, tutorials, numpy; print('AGX OK')"
```

If the import check prints `AGX OK`, the environment is wired up correctly.
