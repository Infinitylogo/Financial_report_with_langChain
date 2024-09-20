#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define default values
HOST="0.0.0.0"
PORT=5000
ENV="development"

# Print a message
echo "Starting Quart app..."

# Export environment variables
export QUART_APP=app:app  # app refers to your app instance in the main file (if it's named `app.py`)
export QUART_ENV=$ENV

# Create uploads folder if not exists
if [ ! -d "uploads" ]; then
  mkdir uploads
  echo "Created uploads directory."
fi

# Run the Quart application
quart run --host=$HOST --port=$PORT

# Print a success message
echo "Quart app is running at http://$HOST:$PORT"
