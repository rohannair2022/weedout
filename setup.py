from setuptools import setup, find_packages

setup(
    name='weedout', 
    version='0.1', 
    packages=find_packages(), 
    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'statsmodels',
        'imbalanced-learn',
        'scipy',
        'tqdm',
        'typing'
    ],
)