# agx_wrap/world.py
import math
import numpy as np
import agx
import agxSDK
import agxCollide
import agxOSG
import agxRender
import agxModel

from agxPythonModules.utils.environment import simulation, root, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array


def _color(node, rgba):
    agxOSG.setDiffuseColor(node, agx.Vec4f(*rgba)) # RGBA
    agxOSG.setAmbientColor(node, agx.Vec4f(1)) # White ambient
    agxOSG.setSpecularColor(node, agx.Vec4f(1)) # White specular
    agxOSG.setShininess(node, 120) # Shininess


# Ocean creation utility
def create_ocean(height=1.5, res_xy=(100, 100), size_xy=(1000, 1000), cell_h=20.0):
    """
    Builds animated heightfield-water + hydrodynamics registration (WindAndWaterController).
    Returns (hf, water_geom, wwc, wave_updater).
    """
    # Heightfield
    rx, ry = res_xy # Resolution
    sx, sy = size_xy # Size
    hf = agxCollide.HeightField(rx, ry, sx, sy, cell_h) # cell size in Z
    water_geom = agxCollide.Geometry(hf) # Water geometry
    water_geom.setMaterial(agx.Material("waterMaterial")) # Water material
    node = agxOSG.createVisual(water_geom, root()) # Visual node
    _color(node, (0.0, 0.74902, 1.0, 0.5)) # Water color (RGBA)

    simulation().add(water_geom) # Add water geometry to simulation

    # Hydrodynamics controller (water registration)
    wwc = agxModel.WindAndWaterController() # Hydrodynamics controller
    wwc.addWater(water_geom) # Register water geometry
    simulation().add(wwc) # Add controller to simulation

    # Pre-allocate heights vector and wrap as numpy
    heights = agx.RealVector(hf.getResolutionX() * hf.getResolutionY()) # Pre-allocated heights vector
    # Initialize heights to zero
    for _ in range(hf.getResolutionX() * hf.getResolutionY()):
        heights.append(0.0)

    # Wrap heights as numpy array for easy manipulation
    np_heights = wrap_vector_as_numpy_array(heights, np.float64).reshape(
        (hf.getResolutionX(), hf.getResolutionY())
    )

    # Build indices once
    jj = np.stack((np.arange(hf.getResolutionY()),) * hf.getResolutionX()) # Y indices
    ii = np.stack((np.arange(hf.getResolutionX()),) * hf.getResolutionY(), axis=1) # X indices

    # Wave parameters
    amp = float(height)

    # Wave function
    def wave_function(t: float):
        # Two-component traveling wave (tunable):
        np_heights[:] = amp * (
            0.40 * np.sin(1.00 * jj + 0.60 * t) +
            0.10 * np.sin(1.20 * ii + 0.60 * jj + 1.45 * t)
        ) 
        hf.setHeights(heights) # Update heightfield heights

    # Hook wave as PRE_COLLIDE so contacts see updated surface
    Sec.preCollideCallback(lambda t: wave_function(t)) # Pre-collide callback for wave update
    wave_function(0.0) # Initial wave update

    # Camera (optional nice default)
    cam = application().getCameraData() # Get camera data
    cam.eye = agx.Vec3(31.1, -87.59, 44.283) # Camera eye position
    cam.center = agx.Vec3(9.57, 2.147, 6.95) # Camera center position
    cam.up = agx.Vec3(-0.039, 0.375, 0.925) # Camera up vector
    cam.nearClippingPlane = 0.1 # Near clipping plane
    cam.farClippingPlane = 5000 # Far clipping plane
    application().applyCameraData(cam) # Apply camera data

    return hf, water_geom, wwc # Return heightfield, water geometry, and wind-and-water controller


# Body colorization utility
def colorize_body(rb: agx.RigidBody, rgba=(1.0, 1.0, 0.8, 1.0)):
    node = agxOSG.createVisual(rb, root()) # Create visual node for rigid body
    _color(node, rgba) # Apply color to the visual node
    
    
