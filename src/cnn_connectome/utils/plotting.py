import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def use_static_flyvis_colorbar(animation):
    """Keep FlyVis's initial colorbar instead of recreating it every frame.

    FlyVis 1.1.3's ``HexScatter.animate`` removes its custom colorbar axes on
    every frame. That removal is incompatible with Matplotlib 3.11. Wrapping
    the instance's animate method leaves the correctly normalized colorbar
    created by ``init`` in place while the hexagon colors continue to update.
    """
    if getattr(animation, "_uses_static_colorbar", False):
        return animation
    if not hasattr(animation, "cbar"):
        raise TypeError("Expected a FlyVis animation with a 'cbar' attribute")

    animate = animation.animate

    def animate_with_static_colorbar(frame):
        show_colorbar = animation.cbar
        animation.cbar = False
        try:
            return animate(frame)
        finally:
            animation.cbar = show_colorbar

    animation.animate = animate_with_static_colorbar
    animation._uses_static_colorbar = True
    return animation