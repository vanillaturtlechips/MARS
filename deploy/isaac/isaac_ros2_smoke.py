"""
Step 1a — Isaac Sim 5.1 ↔ ROS2 bridge smoke test (headless).

Goal: prove the isaacsim.ros2.bridge extension actually publishes to ROS2
BEFORE we build the full warehouse + iw_hub drive graph. It loads a minimal
scene and publishes /clock via an OmniGraph ROS2 clock node.

Run (RunPod, isaac_venv311 = Python 3.11):
    source /opt/ros/humble/setup.bash
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    source /workspace/isaac_venv311/bin/activate
    cd /workspace/MARS
    python deploy/isaac/isaac_ros2_smoke.py

In a second shell (source ROS2 Humble):
    ros2 topic list                 # expect /clock, /rosout, /parameter_events
    ros2 topic echo /clock --once   # expect a rosgraph_msgs/Clock with rising time

If /clock ticks, the bridge works end-to-end and we move to the warehouse scene.
"""
from isaacsim import SimulationApp

# Must be created before importing any other isaac/omni modules.
simulation_app = SimulationApp({"headless": True})

import carb  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# Enable the ROS2 bridge extension, then let kit process it.
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# OmniGraph: tick -> publish ROS2 /clock from the sim clock.
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ],
    },
)

import omni.timeline  # noqa: E402

world.reset()

# CRITICAL: OnPlaybackTick only fires while the timeline is PLAYING. Without
# this the ROS2 publisher node exists (topic is advertised, publisher count=1)
# but never executes, so zero messages are sent and `ros2 topic echo` hangs.
timeline = omni.timeline.get_timeline_interface()
timeline.play()

carb.log_warn("[smoke] timeline playing; publishing /clock. Ctrl+C to stop.")

try:
    while simulation_app.is_running():
        # render=True pumps the action graph each frame (OnPlaybackTick).
        world.step(render=True)
except KeyboardInterrupt:
    pass
finally:
    timeline.stop()
    simulation_app.close()
