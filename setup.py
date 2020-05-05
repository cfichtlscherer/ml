import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml",
    version="0.0.1",
    author="Lennard Benecke, Christopher Fichtlscherer, Ali Raza Khan, Balint Mate, Samanyu Sanjay",
    author_email="fichtlscherer@mailbox.org",
    description="The Package for the Machine Learning Project 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=
        [
        "matplotlib",
        "scipy",
        "tqdm"
        ],
    extras_require=
        {
        "test": [
            "pytest",
            "coverage",
            "pytest-cov",
            "pytest-flake8"]
        },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
