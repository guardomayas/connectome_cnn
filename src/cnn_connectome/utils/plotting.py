import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from IPython.display import Video, display
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

def show_hdr(img, ax=None, mode="log", lo=1, hi=99.5, title=None,
             extent=None, horizon_row=None):
    """Display-only tone map. Input: raw or scaled linear radiance."""
    x = np.asarray(img, dtype=np.float64)
    pos = x[x > 0]
    floor = np.percentile(pos, 0.1) if pos.size else 1e-8

    if mode == "log":
        y = np.log10(np.maximum(x, floor))
    elif mode == "gamma":
        y = np.power(np.maximum(x, 0) / np.percentile(x, hi), 1 / 2.2)
    else:
        y = x

    vmin, vmax = np.percentile(y, [lo, hi])
    ax = ax or plt.gca()
    ax.imshow(y, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal", extent=extent)
    if horizon_row is not None:
        ax.axhline(horizon_row, color="tab:red", lw=0.5)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    return ax

def sample_animation(dataset, idx=0, mode='log', lo=1, hi=99.5, viz_fps=None, title=None, save_path=None):
    """
    Animate one sample from GetNaturalMovies.

    Parameters
    ----------
    dataset : GetNaturalMovies
    idx : int
        Movie index (into dataset.all_movies / dataset.movies)
    mode : {'log', 'linear'}
        'log' displays log10(luminance), contrast-stretched between the
        `lo`/`hi` percentiles. 'linear' displays raw luminance the same way.
    lo, hi : float
        Percentiles (0-100) used to set vmin/vmax for display contrast.
    viz_fps : int or None
        Playback frame rate for the animation
    title : str or None
        Custom title
    save_path : str or None
        If given, save animation to mp4/gif

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """

    movie = dataset.all_movies[idx]  # (T, H, W)

    # dataset.vel_traces is indexed by (img_idx, trace_idx), not by movie
    # idx directly -- movie idx also encodes a phase, so we have to look up
    # which (img_idx, trace_idx) this movie corresponds to.
    img_idx, trace_idx, phase_idx = dataset.movies[idx]
    vel = dataset.vel_traces[img_idx, trace_idx]  # (T,)

    # handle movie format
    if movie.ndim == 4:
        # (T, 1, H, W) -> (H, W, T)
        movie_hw_t = np.moveaxis(movie[:, 0, :, :], 0, 2)
    elif movie.ndim == 3:
        # (T, H, W) -> (H, W, T)
        movie_hw_t = np.moveaxis(movie, 0, 2)
    else:
        raise ValueError(f"Unexpected movie shape: {movie.shape}")

    H, W, T = movie_hw_t.shape
    t = np.arange(T) / dataset.sampleFreq

    # dataset.W is the width of the full (unrendered) equirectangular pano,
    # which spans a full 360 deg in azimuth -- gives deg/pixel. Elevation
    # (the H axis) is never resampled from that same pano, so it shares the
    # same deg/pixel scale.
    deg_per_pix = 360.0 / dataset.W
    az_extent = (W / 2) * deg_per_pix
    el_extent = (H / 2) * deg_per_pix
    
    print(f"Movie extends ({az_extent:.2f}, {el_extent:.2f} degs)")


    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.4])

    ax_img = fig.add_subplot(gs[0])   # spans both rows
    ax_vel = fig.add_subplot(gs[1])   # bottom-right

    positive = movie_hw_t[movie_hw_t > 0]
    floor = np.percentile(positive, 0.1) if positive.size else 1e-8

    if mode == "log":
        y = np.log10(np.maximum(movie_hw_t, floor))
    else:
        y = movie_hw_t

    vmin, vmax = np.percentile(y, [lo, hi])
    im = ax_img.imshow(
        y[:, :, 0], cmap="gray", vmin=vmin, vmax=vmax,
        extent=[-az_extent, az_extent, -el_extent, el_extent],
    )
    ax_img.set_title(f"frame 1/{T}")
    ax_img.set_xlabel("Azimuth (deg)")
    ax_img.set_ylabel("Elevation (deg)")
    if title is not None:
        fig.suptitle(title)


    ax_vel.plot(t, vel, label="v")
    vel_cursor = ax_vel.axvline(t[0], color="k", linestyle="--", alpha=0.7)
    ax_vel.set_title("Velocity trace")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_ylabel("velocity (deg/s)")
    # ax_vel.grid(True, alpha=0.3)
    # ax_vel.legend()

    plt.tight_layout()
    
    def update(frame):
        im.set_data(y[:, :, frame])
        ax_img.set_title(f"frame {frame + 1}/{T}")
        vel_cursor.set_xdata([t[frame], t[frame]])
        return im, vel_cursor

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / viz_fps, blit=False)

    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, fps=viz_fps, dpi=150)

    plt.close(fig)  # close the figure to avoid displaying static version in notebooks
    return anim