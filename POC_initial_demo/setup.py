from setuptools import setup, find_packages

setup(
    name="loophole",
    version="0.1.0",
    description="Automated Lifting of Legacy Scalar Loop Nests into MLIR Tensor Dialects",
    author="LoopHole Project",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "z3-solver>=4.12.0",
        "sympy>=1.12",
        "rich>=13.0",
        "click>=8.1",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
    },
    entry_points={
        "console_scripts": [
            "loophole=loophole.cli:main",
        ],
    },
)
