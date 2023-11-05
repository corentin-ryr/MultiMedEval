from setuptools import setup, find_packages

setup(
    name='MultiMedBench',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'my_script=my_package.my_module:main'
        ]
    }
)