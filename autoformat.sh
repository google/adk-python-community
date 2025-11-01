#!/bin/bash
#
# This script runs the auto-formatters (isort and pyink)
# to fix code style and import order.

set -e

if ! command -v isort &> /dev/null
then
    echo "isort not found, refer to CONTRIBUTING.md to set up dev environment first."
    exit
fi

if ! command -v pyink &> /dev/null
then
    echo "pyink not found, refer to CONTRIBUTING.md to set up dev environment first."
    exit
fi


echo "Running isort to sort imports..."
isort .

echo "Running pyink to reformat code..."
pyink .

echo "Formatting complete."