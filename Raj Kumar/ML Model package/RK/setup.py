import setuptools

setuptools.setup(
    name='rk',
    version='0.4',
    author='Raj Kumar',
    author_email='krajkumar2403@gmail.com',
    description="",
    packages=setuptools.find_packages(),
    install_requires=['numpy','scipy','pandas','imbalanced-learn','scikit-learn'],
    classifiers=[
        "Programming Language::Python::3",
        "License::OSI Approved::MIT Licensed",
        "Operating System:: OS Independent",
    ],
)