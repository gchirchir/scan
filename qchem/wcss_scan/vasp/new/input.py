#!/usr/bin/env python3
"""Simple per-metal entry point: run as `python3 input.py` inside a metal folder."""

from pathlib import Path
import subprocess
import sys

CONFIG = {
    "Al": (4.02, 8, 8, 4, 30),
    "Pt": (3.95, 8, 8, 4, 30),
    "Cu": (3.56, 8, 8, 4, 30),
    "Rh": (3.81, 8, 8, 4, 30),
    "Au": (4.10, 8, 8, 4, 30),
    "Pd": (3.92, 8, 8, 4, 30),
    "Ti": (4.06, 8, 8, 4, 30),
}

here = Path(__file__).resolve().parent
metal = here.name
if metal not in CONFIG:
    raise SystemExit(f"Run this input.py inside one of: {', '.join(CONFIG)}")

root = here.parent
driver = root / "code" / "22_surface_density_dcscan.py"
a, layers, kxy, kz, cutoff = CONFIG[metal]
tag = f"{metal}_L{layers}_k{kxy}_c{cutoff}"
result = root / "code" / "results" / f"surface22_{tag}.json"

if result.is_file() and result.stat().st_size:
    print(f"Already complete: {result}")
    raise SystemExit(0)

cmd = [
    sys.executable, "-u", str(driver), "run",
    "--system", metal,
    "--a", str(a),
    "--layers", str(layers),
    "--kxy", str(kxy),
    "--kz", str(kz),
    "--cutoff", str(cutoff),
]
print("Running:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

