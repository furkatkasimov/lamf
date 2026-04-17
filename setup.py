from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lamf",
    version="0.1.0",
    description="Liquid Audience Measurement Framework — reference implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Furkat Kasimov",
    license="MIT",
    packages=find_packages(exclude=["tests", "scripts", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.10",
        "scikit-posthocs>=0.7.0",
        "matplotlib>=3.6",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
