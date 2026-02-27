import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Video, HTML

def create_simulation_video(sim, history_positions, num_particles, position_scale=0.8, fps=60, filename='simulation.mp4'):
    """
    Creates a video of the true simulation using the TestSim style.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Pre-calculate plot limits to keep the camera still
    all_pos = history_positions
    
    step_size = 1
    all_pos = all_pos[::step_size]
    
    ax.set_xlim(0 - 0.05, position_scale + 0.05)
    ax.set_ylim(0 - 0.05, position_scale + 0.05)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    title_str = f'Simulation ({num_particles} particles)'
    ax.set_title(title_str)
    
    # Calculate dynamic markersize in points to match physical particle size
    particle_diameter = getattr(sim, 'smoothing_length', getattr(sim, 'softening_length', 0.0015))
    p1 = ax.transData.transform((0, 0))
    p2 = ax.transData.transform((particle_diameter, 0))
    marker_size_pt = (p2[0] - p1[0]) * 72.0 / fig.dpi /2
    ax.grid(False)
    
    # Create a gradient from blue to deep blue (ocean-like)
    water_colormap = colors.LinearSegmentedColormap.from_list('ocean_blue', ['#000033', '#0088cc'])
    
    colors_array = np.arange(num_particles) # Gradient by particle index to track motion
    points = ax.scatter(all_pos[0, :, 0], all_pos[0, :, 1], c=colors_array, cmap=water_colormap, s=marker_size_pt**2)
    
    def init():
        points.set_offsets(all_pos[0, :, :2])
        return points,
    
    def update(frame):
        points.set_offsets(all_pos[frame, :, :2])
        return points,
    
    num_frames = all_pos.shape[0]
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    plt.close(fig) # Prevent static plot from showing
    
    # Save animation efficiently directly to mp4 using the FFMpegWriter
    writer = FFMpegWriter(fps=fps, codec='h264', bitrate=2000, extra_args=['-preset', 'ultrafast', '-tune', 'fastdecode'])
    ani.save(filename, writer=writer)
    return Video(filename, embed=True, html_attributes='controls autoplay loop')

def create_rollout_video(gt_positions, pred_positions, num_particles, T, sim_dt=0.0025, bounds=[[0.1, 0.9], [0.1, 0.9]], target_fps=60, filename='rollout.mp4'):
    """
    Creates a side-by-side rollout comparison using the Visualize_wcsph style.
    """
    # Determine plot bounds from ground truth
    if bounds is not None:
        x_lim = bounds[0]
        y_lim = bounds[1]
    else:
        margin = 0.1
        x_lim = [gt_positions[:, :, 0].min() - margin, gt_positions[:, :, 0].max() + margin]
        y_lim = [gt_positions[:, :, 1].min() - margin, gt_positions[:, :, 1].max() + margin]

    # Animation at 60fps with accurate simulation time
    sim_fps = 1.0 / sim_dt               # simulation frames per second of real time
    frame_skip = max(1, int(sim_fps / target_fps))  # skip to match real-time playback
    frames = list(range(0, T, frame_skip))
    interval_ms = 1000.0 / target_fps    # ~16.67ms per frame

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('WCSPH Rollout: Ground Truth vs GNN Prediction', fontsize=14, fontweight='bold')

    for ax, title in [(ax_gt, 'Ground Truth (WCSPH)'), (ax_pred, 'GNN Prediction')]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_facecolor('#ffffff')

    water_colormap = LinearSegmentedColormap.from_list('ocean_blue', ['#0088cc', '#000033'])
    water_colormap_gnn = LinearSegmentedColormap.from_list('ocean_purple', ['#e175ff', '#5a0778'])
    particle_colors = np.arange(num_particles)

    scatter_gt = ax_gt.scatter(gt_positions[0, :, 0], gt_positions[0, :, 1], s=4, c=particle_colors, cmap=water_colormap, alpha=0.9)
    scatter_pred = ax_pred.scatter(pred_positions[0, :, 0], pred_positions[0, :, 1], s=4, c=particle_colors, cmap=water_colormap_gnn, alpha=0.9)
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11)

    def init():
        scatter_gt.set_offsets(gt_positions[0, :, :2])
        scatter_pred.set_offsets(pred_positions[0, :, :2])
        return scatter_gt, scatter_pred, time_text

    def animate(frame_idx):
        t = frames[frame_idx]
        scatter_gt.set_offsets(gt_positions[t, :, :2])
        scatter_pred.set_offsets(pred_positions[t, :, :2])
        time_text.set_text(f't = {t * sim_dt:.3f}s  (step {t}/{T-1})')
        return scatter_gt, scatter_pred, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames), interval=interval_ms, blit=True)
    plt.close(fig)
    
    writer = FFMpegWriter(fps=target_fps, codec='h264', bitrate=2000, extra_args=['-preset', 'ultrafast', '-tune', 'fastdecode'])
    anim.save(filename, writer=writer)
    return Video(filename, embed=True, html_attributes='controls autoplay loop')
