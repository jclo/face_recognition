#!/bin/sh
# ******************************************************************************
# A bash script to install Python modules in the project environment required
# by the Atom's packages: Hydrogen, linter-flake8 and linter-docstyle.
# ******************************************************************************
pip install --upgrade \
  jupyter \
  ipykernel \
  'python-language-server[all]'
