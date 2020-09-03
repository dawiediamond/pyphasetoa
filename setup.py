#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'matplotlib', 'numpy', 'scipy', 'tqdm']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="David Hercules Diamond",
    author_email='dawie.diamond@up.ac.za',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python package used to calculate the ToA of a blade at a sensor using the instantaneous phase of a proximity probe signal.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyphasetoa',
    name='pyphasetoa',
    packages=find_packages(include=['pyphasetoa', 'pyphasetoa.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dawiediamond/pyphasetoa',
    version='0.1.0',
    zip_safe=False,
)
