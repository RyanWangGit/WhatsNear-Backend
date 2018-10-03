from setuptools import setup, find_packages

# Get the long description from the relevant file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='WhatsNear-Backend',
    version='0.1',
    description='WhatsNear-Backend - Web App That Ranks Candidate Store Locations Using RankNet.',
    long_description=long_description,
    url='',
    author='Ryan Wang',
    author_email='ryanwang.cs@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Neural Network',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Neural Network',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'python-geohash', 'tensorflow', 'tornado', 'h5py', 'haversine', 'progress'],
    extras_requires={
        'test': ['pytest-cov', 'pytest', 'coverage'],
    },
    entry_points={
        'console_scripts': [
            'whatsnear=whatsnear.__main__:main',
        ],
    },
)
