from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'PackMetric'
LONG_DESCRIPTION = 'A lightweight collection for easy batch updates of torchmetrics, ' \
                   'designed to integrate seamlessly with PyTorch Lightning.'

# Setting up
setup(
    name="packmetric",
    version=VERSION,
    author="e_yi",
    url="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['lightning>=2', 'torch>=2.0.0'],
)
