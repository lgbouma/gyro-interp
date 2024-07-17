# -*- coding: utf-8 -*-
"""
setup.py - boilerplate
"""
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)

def readme():
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy>=1.7.0',
    'pandas>=1.2.4',
    'astropy>=4.0.4',
    'matplotlib',
]

EXTRAS_REQUIRE = {
    'all':[
        'cdips',
    ]
}

###############
## RUN SETUP ##
###############

# run setup.
version = 0.5
setup(
    name='gyrointerp',
    version=version,
    description=(
        "Gyrochronology via interpolation of open cluster rotation sequences."
    ),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    keywords='astronomy',
    url='https://github.com/lgbouma/gyro-interp',
    download_url=f'https://github.com/lgbouma/gyro-interp/archive/refs/tags/v{str(version).replace(".","")}.tar.gz',
    author='Luke Bouma',
    author_email='bouma.luke@gmail.com',
    license='MIT',
    packages=[
        'gyrointerp',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest==3.8.2',],
    cmdclass={'test':PyTest},
    include_package_data=True,
    zip_safe=False,
)
