import setuptools

setuptools.setup(
    name='dj',
    version='0.0.2',
    author='Dhamu Pradeep',
    author_email='dhamupradeep2610@gmail.com',
    description='ridge,nb,xgb',
    packages=setuptools.find_packages(),
    install_requires = ['numpy','scipy',],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
)