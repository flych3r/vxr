"""Setup package."""

from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    DEPENDENCIES = f.readlines()

LICENSE = 'MIT License'

CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
]
if LICENSE:
    CLASSIFIERS.append(f'License :: OSI Approved :: {LICENSE}')

setup(
    name='vxr',
    version='0.0.1',
    author='Matheus Xavier Sampaio',
    author_email='matheus.sampaio011@gmail.com',
    license=LICENSE,
    python_requires='>=3.7',
    description='Generate chest X-ray reports using Vision Transformers',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/flych3r/vxr',
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    install_requires=DEPENDENCIES,
    include_package_data=True
)
