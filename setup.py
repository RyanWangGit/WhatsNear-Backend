from setuptools import setup, find_packages

# Get the long description from the relevant file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='RankNear',
    version='0.1',
    description='RankNear - Rank the locations nearby using RankNet.',
    long_description=long_description,
    url='https://github.com/RyanWangGit/ranknear',
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
    install_requires=['numpy', 'pygeohash', 'tensorflow', 'haversine', 'progress'],
    extras_requires={
        'test': ['pytest-cov', 'pytest', 'coverage'],
    },
    entry_points={
        'console_scripts': [
            'ranknear=ranknear.__main__:main',
        ],
    },
)
