#!/bin/bash
# Rayleigh Duplicate Detector — Mac Launcher
# Double-click this file to start the app
cd "$(dirname "$0")"
echo "Installing dependencies..."
pip3 install -q streamlit numpy scipy matplotlib pandas 2>/dev/null
echo "Starting Rayleigh Duplicate Detector..."
streamlit run app.py
