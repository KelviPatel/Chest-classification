import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%()s')

project_name='cnnclassifer'
 # all the files needed
files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/componenets/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for filepath in files:
    filepath=Path(filepath) # somethime it may through error with paths directly specified therefore "Path" is used
    filedir,filename=os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directoty; {filedir} for the file : {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
        logging.info(f"Creating empty file : {filepath}")

    else :
        logging.info(f"{filename} already exists ")
