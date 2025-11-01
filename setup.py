from setuptools import find_packages, setup

setup(
    name="susurrus",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.0.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "requests>=2.25.1",
        "pydub>=0.25.1",
    ],
    entry_points={
        "console_scripts": [
            "susurrus=main:main",
        ],
    },
)
