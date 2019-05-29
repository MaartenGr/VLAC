import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vlac',
    author='Maarten Grootendorst',
    author_email='maartengrootendorst@gmail.com',
    url="https://github.com/MaartenGr/VLAC",
    package_dir={'vlac': 'vlac'},
    packages=find_packages(),
    description='Tool for creating document features',
    install_requires=['numpy', 'sklearn'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.1.2.5',
    license='MIT',
)