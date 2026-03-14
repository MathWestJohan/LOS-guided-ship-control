# main.py
import agx
from agxPythonModules.utils.environment import init_app
from runtime.runner import build_scene_and_start

# Launch using AGX's application wrapper. We let our runner build scene & callbacks.
init = init_app(
    name=__name__,
    scenes=[(build_scene_and_start, "M")],
    autoStepping=True
)
