#!/bin/bash
python3 -m venv venv_nlp 
source venv_nlp/Scripts/activate
pip3 install --upgrade pip
pip install -r requirements.txt
chmod 755 setup.sh #rwx->u and rx->g/o
