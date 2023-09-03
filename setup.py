from setuptools import setup,find_packages
from typing import List

hypen="-e ."
def get_requirement(file_path:str)->List[str]:
    """This function will return all the requirements"""
    requirements=[]
    with open (file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    if hypen in requirements:
        requirements.remove(hypen)
    return requirements

setup (name="PDF Reading",version="0.0.1",author="Pankaj Bhatia",author_email="pankaj.bhatia85@outlook.com",
       install_requirs=get_requirement("requirements.txt"),packages=find_packages())


