import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DISAtool",
    version="1.0.0",
    author="L. Alexandre, R.S. Costa, R. Henriques",
    author_email="leonardoalexandre@tecnico.ulisboa.pt",
    description="A library used to assess the informative and discriminative properties of subspaces/patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["multi-item discretization", "prior-free discretization", "heterogeneous biological data", "data mining"],
    url="https://github.com/JupitersMight/DISA",
    project_urls={
        "Bug Tracker": "https://github.com/JupitersMight/DI2/issues",
    },
    packages=['DISAtool'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
