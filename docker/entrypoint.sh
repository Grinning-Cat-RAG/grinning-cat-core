#!/bin/bash
set -e

echo "--- 🐱 Grinning Cat Boot Sequence ---"

echo "Running migrations..."
# the venv is on the PATH: `uv run` would sync (and build) the project at every start, which needs the network
python migrations/manage_migrations.py upgrade head

echo "--- 🐱 Starting Grinning Cat ---"

exec "$@"