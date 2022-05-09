from setuptools import setup, find_packages

setup(
   name='upright',
   version='0.0.1',
   description='Upright Orientation Estimation',
   author='Daniel Swoboda, Kanghyun Kim',
   packages=find_packages(exclude=["data", "scripts"]),  #same as name
)