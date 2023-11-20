from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='MultiMedBench',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'my_script=my_package.my_module:main'
        ]
    }
)