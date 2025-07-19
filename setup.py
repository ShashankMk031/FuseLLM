from setuptools import setup, find_packages

setup(
    name="fusellm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "gradio>=3.0.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "soundfile>=0.10.0",
        "librosa>=0.8.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "isort>=5.0.0",
            "mypy>=0.9.0",
            "pylint>=2.12.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "fusellm=main:main",
        ],
    },
)
