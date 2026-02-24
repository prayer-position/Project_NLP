#!/bin/bash
python3 -m venv venv_nlp 
source venv_nlp/Scripts/activate
pip3 install --upgrade pip
pip install -r requirements.txt
<<<<<<< HEAD

=======
>>>>>>> 29d1af0f55e92611661ca5585fafc6805986fbba
chmod 755 setup.sh #rwx->u and rx->g/o
