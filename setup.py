from setuptools import setup

setup(
    name='machinemode',
    packages=["machinemode"],
    entry_points={
        "console_scripts": ["machinemode = machinemode.__main__:main"]},
    install_requires=[],
)
