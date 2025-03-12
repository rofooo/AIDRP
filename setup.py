from setuptools import setup, find_packages

setup(
    name="aidrp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "networkx>=3.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scapy>=2.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Based Dynamic Routing Protocol",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aidrp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 