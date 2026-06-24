"""Diagnostic: what prim type/path does LidarRtx actually create?
Run: source deploy/isaac/env_isaac.sh && python deploy/isaac/diag_lidar.py
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.ros2.bridge")
app.update()

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.sensors.rtx import LidarRtx  # noqa: E402

LidarRtx(
    prim_path="/World/lidar",
    name="l",
    position=np.array([0.0, 0.0, 0.3]),
    config_file_name="RPLIDAR_S2E",
)
app.update()

st = omni.usd.get_context().get_stage()
print("==== prims containing 'lidar' ====")
for p in st.Traverse():
    s = str(p.GetPath())
    if "lidar" in s.lower():
        print("PRIM:", p.GetTypeName(), s)
print("==================================")

app.close()
