#!/bin/bash

# Removes all .pyc and .pyo files, and __pycache__ directories.
find . -type f -name "*.py[co]" -delete
find . -type d -name "__pycache__" -delete
find . -type f -name ".DS_Store" -delete

# Removes build and *.egg-info directories
rm -rf build/
rm -rf *.egg-info
