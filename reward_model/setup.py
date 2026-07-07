from setuptools import setup, find_packages

setup(
    name="reward_model",
    version="0.1.0",
    description="Reward model trained from scratch using Bradley-Terry pairwise ranking loss.",
    author="Maninder Singh",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
)