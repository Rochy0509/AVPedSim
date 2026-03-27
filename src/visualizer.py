import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pedestrian_model   import PedestrianManager
from sensor_fusion      import SensorFusion
from vehicle_controller import VehicleController, compute_ttc


# Parameters

SIM_DURATION     = 60.0   # seconds for this visual trial
DT               = 0.1
VEHICLE_START    = -30.0
DRAW_EVERY       = 3      # redraw every N ticks (lower = smoother but slower)
SEED             = 42

CRUISE_SPEED     = 8.33   # m/s — must match vehicle_controller


ROAD_LEFT        = -4.0
ROAD_RIGHT       =  4.0
CROSSWALK_X_MIN  = 10.0
CROSSWALK_X_MAX  = 15.0

SIDEWALK_WIDTH   = 3.0    # visual width of sidewalk strips


STATE_COLORS = {
    "WAITING":  "steelblue",
    "CROSSING": "limegreen",
    "RUNNING":  "red",
    "WALKING":  "mediumpurple",   # sidewalk walkers
}

# Camera FOV colours (one per camera in SensorFusion)
CAMERA_COLORS = [
    "#00bfff",   # front main — cyan
    "#ff6347",   # front wide — tomato
    "#ffd700",   # front telephoto — gold
    "#32cd32",   # B-pillar left — lime
    "#da70d6",   # B-pillar right — orchid
]


def setup_figure():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-35, 40)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("Road axis (m)")
    ax.set_ylabel("Across road (m)")
    return fig, ax


def draw_scene(ax, manager, fusion, controller, sim_time, estimates):

    ax.cla()
    ax.set_xlim(-35, 40)
    ax.set_ylim(-10, 10)
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.set_xlabel("Road axis (m)", color="white")
    ax.tick_params(colors="white")

    # Bottom sidewalk (below road)
    ax.add_patch(patches.Rectangle(
        (ax.get_xlim()[0], ROAD_LEFT - SIDEWALK_WIDTH),
        ax.get_xlim()[1] - ax.get_xlim()[0],
        SIDEWALK_WIDTH,
        color="#2a2a3e", zorder=0
    ))
    # Top sidewalk (above road)
    ax.add_patch(patches.Rectangle(
        (ax.get_xlim()[0], ROAD_RIGHT),
        ax.get_xlim()[1] - ax.get_xlim()[0],
        SIDEWALK_WIDTH,
        color="#2a2a3e", zorder=0
    ))

    # Road
    road = patches.Rectangle(
        (ax.get_xlim()[0], ROAD_LEFT),
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ROAD_RIGHT - ROAD_LEFT,
        color="#444444"
    )
    ax.add_patch(road)

    # Centre dashes
    dash_len = 2.0
    gap = 2.0
    x = ax.get_xlim()[0]
    while x < ax.get_xlim()[1]:
        ax.plot([x, x + dash_len], [0, 0], color="yellow",
                linewidth=0.8, alpha=0.5, zorder=1)
        x += dash_len + gap

    # Crosswalk stripes
    stripe_width = 0.8
    y = ROAD_LEFT
    while y < ROAD_RIGHT:
        ax.add_patch(patches.Rectangle(
            (CROSSWALK_X_MIN, y),
            CROSSWALK_X_MAX - CROSSWALK_X_MIN,
            stripe_width,
            color="white", alpha=0.15
        ))
        y += stripe_width * 2

    # Crosswalk outline
    ax.add_patch(patches.Rectangle(
        (CROSSWALK_X_MIN, ROAD_LEFT),
        CROSSWALK_X_MAX - CROSSWALK_X_MIN,
        ROAD_RIGHT - ROAD_LEFT,
        fill=False, edgecolor="white", linewidth=1.5, linestyle="--"
    ))

    # Road edges
    ax.axhline(ROAD_LEFT,  color="white", linewidth=1.5)
    ax.axhline(ROAD_RIGHT, color="white", linewidth=1.5)

    # LiDAR range circle
    lidar_range = fusion.lidar.max_range
    lidar_circle = patches.Circle(
        (controller.x, 0.0), lidar_range,
        fill=False, edgecolor="cyan", linewidth=0.8,
        linestyle=":", alpha=0.5, zorder=3
    )
    ax.add_patch(lidar_circle)
    ax.text(controller.x + lidar_range + 0.3, 0.3,
            f"LiDAR {lidar_range:.0f}m",
            color="cyan", fontsize=6, alpha=0.7)

    # Camera FOV wedges
    for i, cam in enumerate(fusion.cameras):
        color = CAMERA_COLORS[i % len(CAMERA_COLORS)]
        # Wedge center angle: heading converted so 0° = right
        center_angle_deg = -np.rad2deg(cam.heading_rad)
        fov_deg = np.rad2deg(cam.fov_rad)
        # Clamp range for visibility
        draw_range = min(cam.max_range, 30.0)

        wedge = patches.Wedge(
            (controller.x, 0.0),
            draw_range,
            center_angle_deg - fov_deg / 2,
            center_angle_deg + fov_deg / 2,
            color=color, alpha=0.08, zorder=2
        )
        ax.add_patch(wedge)

        # Draw wedge edges
        for sign in [-1, 1]:
            edge_angle = np.deg2rad(center_angle_deg + sign * fov_deg / 2)
            ex = controller.x + draw_range * np.cos(edge_angle)
            ey = 0.0 + draw_range * np.sin(edge_angle)
            ax.plot([controller.x, ex], [0.0, ey],
                    color=color, linewidth=0.5, alpha=0.35, zorder=2)

    # Vehicle
    car_length = 4.0
    car_width  = 1.8
    car = patches.FancyBboxPatch(
        (controller.x - car_length / 2, -car_width / 2),
        car_length, car_width,
        boxstyle="round,pad=0.1",
        color="gold", zorder=5
    )
    ax.add_patch(car)

    # Headlights
    for side in [-1, 1]:
        ax.plot(controller.x + car_length / 2,
                side * car_width / 4,
                "o", color="white", markersize=3, zorder=6)

    ax.text(
        controller.x, -car_width / 2 - 1.0,
        f"{controller.speed * 3.6:.1f} km/h",
        color="gold", fontsize=7, ha="center"
    )

    # Pedestrians
    for ped in manager.get_active_pedestrians():
        is_jay = getattr(ped, "jaywalker", False)
        is_sw  = getattr(ped, "sidewalk_walker", False)

        if is_jay:
            color  = "#ff7f50"       # coral — distinct jaywalker color
            marker = "s"
            ms     = 8
        elif is_sw:
            color  = STATE_COLORS.get(ped.state, "mediumpurple")
            marker = "o"
            ms     = 6
        else:
            color  = STATE_COLORS.get(ped.state, "white")
            marker = "o"
            ms     = 8

        ax.plot(ped.x, ped.y, marker, color=color,
                markersize=ms, zorder=6)

        label = f"P{ped.id}"
        if is_jay:
            label += f"\nJAY-{ped.state[:3]}"
        elif is_sw:
            label += "\nWLK"
        else:
            label += f"\n{ped.state[:3]}"

        ax.text(ped.x + 0.3, ped.y + 0.4,
                label, color=color, fontsize=6)

    # Kalman estimates
    est_map = {e["pedestrian_id"]: e for e in estimates}
    for pid, e in est_map.items():
        ax.plot(e["x_est"], e["y_est"], "x",
                color="orange", markersize=8,
                markeredgewidth=2, zorder=7)

    # Detection lines
    for e in estimates:
        ax.plot([controller.x, e["x_est"]], [0.0, e["y_est"]],
                color="orange", linewidth=0.4, alpha=0.4, zorder=4)

    # Legend
    legend_items = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=s)
        for s, c in STATE_COLORS.items()
    ]
    legend_items.append(
        plt.Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="limegreen", markersize=8,
                   label="Jaywalker")
    )
    legend_items.append(
        plt.Line2D([0],[0], marker="x", color="orange",
                   markersize=8, markeredgewidth=2, label="KF estimate")
    )
    ax.legend(handles=legend_items, loc="upper right",
              facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    # Title
    n_jay = sum(1 for p in manager.get_active_pedestrians()
                if getattr(p, "jaywalker", False) and p.state in ("CROSSING", "RUNNING"))
    ax.set_title(
        f"t={sim_time:.1f}s  |  "
        f"Vehicle x={controller.x:.1f}m  |  "
        f"Active peds: {len(manager.get_active_pedestrians())}  |  "
        f"Jaywalkers: {n_jay}  |  "
        f"Tracks: {len(estimates)}",
        color="white", fontsize=9
    )

# Main

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    manager    = PedestrianManager()
    fusion     = SensorFusion()
    controller = VehicleController(start_x=VEHICLE_START)

    fig, ax = setup_figure()
    plt.ion()          # interactive mode — lets plt.pause() work
    plt.show()

    sim_time  = 0.0
    tick      = 0
    estimates = []

    print("=== Live Simulation ===  (close window to stop)\n")

    while sim_time < SIM_DURATION:

        vehicle_pos = (controller.x, 0.0)

        # Update pedestrian model
        manager.update(sim_time, DT, vehicle_pos)

        # Sensor fusion
        estimates = fusion.update(vehicle_pos, manager.get_active_pedestrians())

        # Feed estimated TTC back to crossing pedestrians
        est_map  = {e["pedestrian_id"]: e for e in estimates}
        conf_ids = set(est_map.keys())
        for ped in manager.get_crossing_pedestrians():
            if ped.id in conf_ids:
                est_x           = est_map[ped.id]["x_est"]
                ped.current_ttc = compute_ttc(
                    controller.x, controller.speed, est_x
                )

        # Update vehicle
        controller.update(
            [p for p in manager.get_crossing_pedestrians()
             if p.id in conf_ids],
            DT, vehicle_pos
        )

        # Car looping: reset after passing crosswalk
        if controller.x >= CROSSWALK_X_MAX + 5.0:
            controller.x     = VEHICLE_START
            controller.speed = CRUISE_SPEED

        # Redraw every DRAW_EVERY ticks
        if tick % DRAW_EVERY == 0:
            draw_scene(ax, manager, fusion, controller, sim_time, estimates)
            plt.pause(0.001)

        # Stop if window is closed
        if not plt.fignum_exists(fig.number):
            print("Window closed — stopping.")
            break

        sim_time = round(sim_time + DT, 6)
        tick    += 1

    plt.ioff()
    plt.show()
    print("\nDone")