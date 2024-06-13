#!/usr/bin/bash
set -e

python3.9 -m venv --copies venv
source venv/bin/activate

pip3 install ipykernel
pip3 install -r requirements.txt
python -m ipykernel install --user --name=cmdstan_py

deactivate
