from setuptools import setup

with open("requirements.txt") as f:
    requirements = [line for line in f if not line.startswith("#")]
    
setup(
    name='machinemode',
    version="0.1",
    author="Kirill Grevtsov",
    author_email="grevtsovkirill@gmail.com",
    description="build a model for classifying the state of machine using the measurements from the sensors",
    packages=["machinemode"],
    entry_points={
        "console_scripts": ["machinemode = machinemode.__main__:main"]},
    install_requires=requirements,
)
