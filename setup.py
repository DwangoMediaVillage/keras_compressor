from setuptools import find_packages, setup

setup(
    name='keras_compressor',
    version='0.0.1',
    packages=find_packages(
        exclude=['example'],
    ),
    url='',
    license='Apache License v2',
    author='Kosuke Kusano',
    author_email='kosuke_kusano@dwango.co.jp',
    description='',
    install_requires=[
        'numpy',
        'h5py',
        'keras>=2.0.0',
        'scipy',
        'scikit-learn',
    ],
    scripts=['bin/keras-compressor.py'],
)
