"""List Isaac S3 asset folders to see if a real charging-station / dock asset exists
(so we can use it instead of the primitive dock). Asset listing needs NO GPU.

    source deploy/isaac/env_isaac.sh
    python deploy/isaac/list_charger_assets.py
"""
import omni.client

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
FOLDERS = [
    "Isaac/Robots/Idealworks",
    "Isaac/Robots/Idealworks/iwhub",
    "Isaac/Props",
    "Isaac/Environments/Simple_Warehouse/Props",
    "Isaac/Environments/Simple_Warehouse",
]
KW = ("charg", "dock", "station", "battery", "power", "pad")

for f in FOLDERS:
    res, entries = omni.client.list(BASE + f)
    print("==", f, "->", res)
    for e in (entries or []):
        name = e.relative_path
        hit = "   <== candidate" if any(k in name.lower() for k in KW) else ""
        print("    ", name, hit)
