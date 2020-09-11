import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy"]

setuptools.setup(
    name="simulators",
    version="1.0.0",
    author="Ravi Haksar",
    author_email="rhaksar@stanford.edu",
    description="A collection of simulators of graph-based Markov models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rhaksar/simulators",
    packages=setuptools.find_packages(),
    package_data={"simulators": ["epidemics/west_africa_graph.pkl"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    license="MIT License",
)