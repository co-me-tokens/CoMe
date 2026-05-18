#!/bin/bash

# This script cleans up Python cache directories and empty directories in the current path.

echo "Cleaning up __pycache__ and empty directories..."

find -depth -type d -name '__pycache__' -exec rm -r {} \;

until find -type d -empty -exec rm -r {} \; ; do echo "Removing empty directories..."; done

echo "done."
