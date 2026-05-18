from setuptools import setup, find_packages

setup(
    name='salad',
    version='0.1',
    packages=find_packages(include=['salad', 'salad.*']),
)

