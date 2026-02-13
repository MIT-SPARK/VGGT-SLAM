from setuptools import setup, find_packages

setup(
    name='vggt_slam',
    version='2.0.0',
    description='A feedforward SLAM system optimized on the SL(4) manifold.',
    author='Dominic Maggio',
    packages=find_packages(include=['evals', 'evals.*', 'vggt_slam', 'vggt_slam.*']),
)

