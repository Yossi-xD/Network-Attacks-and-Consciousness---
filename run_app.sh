#!/bin/bash
# Launches the Face Morph Studio demo app on Linux.
if [ -d ".venv" ]; then
    .venv/bin/streamlit run src/app.py --server.address=127.0.0.1 --server.port=8501
else
    echo "Virtual environment '.venv' not found. Please run the setup first."
    exit 1
fi
