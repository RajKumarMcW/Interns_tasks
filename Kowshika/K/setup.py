import setuptools

setuptools.setup(
    name='k',
    version='0.4',
    author='Kowshika',
    author_email='kowshikakumar2002@gmail.com',
    description="",
    packages=setuptools.find_packages(),
    install_requires=['numpy','scipy','pandas','imbalanced-learn','scikit-learn'],
    classifiers=[
        "Programming Language::Python::3",
        "License::OSI Approved::MIT Licensed",
        "Operating System:: OS Independent",
    ],
)
