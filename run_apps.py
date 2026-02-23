#!/usr/bin/env python
"""
Launcher for AgriSens ecosystem:
- Static frontend (HTML/JS) on port 8000
- Crop Recommendation Streamlit app on port 8001
- Plant Disease Detection Streamlit app on port 8002
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).parent

# Static site directory (where your index.html lives)
STATIC_DIR = BASE_DIR / "AgriSens-web-app"

# --- Streamlit app paths (update these to your actual files) ---
CROP_APP = BASE_DIR / "crop-recommendation" / "webapp.py"          # 👈 change if needed
DISEASE_APP = BASE_DIR / "plant-disease-identification" / "disease.py"   # 👈 change if needed

# Ports
STATIC_PORT = 8000
CROP_PORT = 8001
DISEASE_PORT = 8002
# =========================

def start_static_server(port):
    """Start Python HTTP server for static files."""
    os.chdir(STATIC_DIR)
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def start_streamlit_app(script_path, port):
    """Start a Streamlit app on given port."""
    return subprocess.Popen(
        ["streamlit", "run", str(script_path),
         "--server.port", str(port),
         "--server.headless", "true",          # avoid auto-opening browser
         "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def main():
    # Check if static directory exists
    if not STATIC_DIR.is_dir():
        print(f"❌ Static directory not found: {STATIC_DIR}")
        sys.exit(1)

    processes = []

    # Start static server
    print(f"\n🚀 Starting static site on port {STATIC_PORT}...")
    try:
        static_proc = start_static_server(STATIC_PORT)
        processes.append(static_proc)
        print(f"   Serving static files from {STATIC_DIR}")
        print(f"   Access at: http://localhost:{STATIC_PORT}")
    except Exception as e:
        print(f"❌ Failed to start static server: {e}")
        sys.exit(1)

    # Start crop app if the file exists
    if CROP_APP.is_file():
        print(f"\n🌽 Starting Crop Recommendation app on port {CROP_PORT}...")
        try:
            crop_proc = start_streamlit_app(CROP_APP, CROP_PORT)
            processes.append(crop_proc)
            print(f"   Access at: http://localhost:{CROP_PORT}")
        except Exception as e:
            print(f"❌ Failed to start crop app: {e}")
    else:
        print(f"\n⚠️  Crop app not found at {CROP_APP} – skipping.")

    # Start disease app if the file exists
    if DISEASE_APP.is_file():
        print(f"\n🌱 Starting Plant Disease Detection app on port {DISEASE_PORT}...")
        try:
            disease_proc = start_streamlit_app(DISEASE_APP, DISEASE_PORT)
            processes.append(disease_proc)
            print(f"   Access at: http://localhost:{DISEASE_PORT}")
        except Exception as e:
            print(f"❌ Failed to start disease app: {e}")
    else:
        print(f"\n⚠️  Disease app not found at {DISEASE_APP} – skipping.")

    if not processes:
        print("\n❌ No services started. Exiting.")
        sys.exit(1)

    print("\n✅ All services started. Press Ctrl+C to stop everything.\n")

    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        for proc in processes:
            proc.terminate()
            proc.wait()
        print("✅ Done.")

if __name__ == "__main__":
    main()
