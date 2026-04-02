Follow if you want to use virtual environmen, otherwise ignore.
Using the terminal.

*1) set up new virtual environment (venv) using:*
python -m venv venv

*2) activate environment using (this must be done at the start of every session):*
source venv/bin/activate

*3) install libraries in venv:*
pip install networkx geopy requests gymnasium

*4) Create snapshot of libraires in venv in a text file called requiremnts and overwirte current file:*
pip freeze > requirements.txt

*5) Deactivate venv when done for the day:*
deactivate
