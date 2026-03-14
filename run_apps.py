#!/usr/bin/env python
"""
AgriSens Ecosystem Launcher with Flask Authentication
------------------------------------------------------
Starts:
- Flask app (login/signup) on port 8000
- Crop Recommendation Streamlit app on port 8001
- Plant Disease Detection Streamlit app on port 8002

Press Ctrl+C to stop all services.
"""

import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

from flask import app, flash, request
from requests import session

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).parent

# --- Flask app settings ---
FLASK_PORT = 8000

# --- Streamlit app paths (update these to your actual files) ---
CROP_APP = BASE_DIR / "crop-recommendation" / "webapp.py"          # 👈 change if needed
DISEASE_APP = BASE_DIR / "plant-disease-identification" / "disease.py"   # 👈 change if needed

# Streamlit ports
CROP_PORT = 8001
DISEASE_PORT = 8002
# =========================

def run_flask(port):
    """Start the Flask app."""
    from flask import Flask, render_template, request, redirect, url_for, flash

    # Tell Flask to look for templates and static files in AgriSens-web-app
    app = Flask(__name__,
                template_folder='AgriSens-web-app',
                static_folder='AgriSens-web-app')
    app.secret_key = 'your-secret-key-here'  # Change in production!

    # Simple in-memory user storage (for demo only)
    users = {}  # username: password

    @app.route('/')
    def index():
        print("Index route accessed")  # Debug
        return render_template('index.html')

    @app.route('/login.html', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if username in users and users[username] == password:
                session['username'] = username          # log the user in
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
        return render_template('login.html')
    
    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        print("Signup route accessed")  # Debug
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            confirm = request.form.get('confirm_password')

            if not username or not password:
                flash('Username and password are required.', 'danger')
            elif username in users:
                flash('Username already exists. Please choose another.', 'danger')
            elif password != confirm:
                flash('Passwords do not match.', 'danger')
            else:
                users[username] = password
                flash('Registration successful! You can now log in.', 'success')
                return redirect(url_for('login'))
        return render_template('signup.html')

    @app.route('/logout')
    def logout():
        session.pop('username', None)
        flash('You have been logged out.', 'info')
        return redirect(url_for('index'))
    
    @app.route('/debug-session')
    def debug_session():
        return f"Session: {session.get('username')}"
    
    from flask import render_template

    @app.route('/explore.html')
    def explore():
        return render_template('explore.html')

    @app.route('/test')
    def test():
        return "Flask is working!"

    print("Flask routes registered.")  # Debug
    # Disable the reloader to avoid multiprocessing issues
    app.run(debug=True, port=port, use_reloader=False)

def start_streamlit_app(script_path, port):
    """Start a Streamlit app as a subprocess."""
    return subprocess.Popen(
        ["streamlit", "run", str(script_path),
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

def main():
    # Check that the Flask templates exist
    templates_dir = BASE_DIR / "AgriSens-web-app"
    if not templates_dir.is_dir():
        print(f"❌ Templates folder not found: {templates_dir}")
        print("   Please ensure the folder exists with index.html, login.html, signup.html")
        sys.exit(1)

    processes = []
    streamlit_procs = []

    # Start Flask in a separate process
    print(f"\n🚀 Starting Flask app on port {FLASK_PORT}...")
    flask_process = multiprocessing.Process(target=run_flask, args=(FLASK_PORT,))
    flask_process.start()
    processes.append(flask_process)
    print(f"   Access at: http://localhost:{FLASK_PORT}")

    # Start Crop app if file exists
    if CROP_APP.is_file():
        print(f"\n🌽 Starting Crop Recommendation app on port {CROP_PORT}...")
        try:
            crop_proc = start_streamlit_app(CROP_APP, CROP_PORT)
            streamlit_procs.append(crop_proc)
            processes.append(crop_proc)
            print(f"   Access at: http://localhost:{CROP_PORT}")
        except Exception as e:
            print(f"❌ Failed to start crop app: {e}")
    else:
        print(f"\n⚠️  Crop app not found at {CROP_APP} – skipping.")

    # Start Disease app if file exists
    if DISEASE_APP.is_file():
        print(f"\n🌱 Starting Plant Disease Detection app on port {DISEASE_PORT}...")
        try:
            disease_proc = start_streamlit_app(DISEASE_APP, DISEASE_PORT)
            streamlit_procs.append(disease_proc)
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
            if isinstance(proc, multiprocessing.Process):
                proc.terminate()
                proc.join()
            else:
                proc.terminate()
                proc.wait()
        print("✅ Done.")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
