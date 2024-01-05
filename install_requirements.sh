#!/bin/bash

# Define the name of the virtual environment
ENV_NAME="vit"

# Create a new virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Check if requirements.txt exists in the current directory
if [ -f "requirements.txt" ]; then
    # Install packages from requirements.txt
    pip install -r requirements.txt

    # List installed packages as a verification step
    pip list

    echo "Packages installed and listed above."

else
    echo "requirements.txt not found in the current directory."
fi

# Deactivate the virtual environment
deactivate

echo "The virtual environment $ENV_NAME has been deactivated."
