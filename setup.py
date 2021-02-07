import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pipeline_transformations",
    author="Stu Day",
    author_email="stuartday274@gmail.com",
    description="A suit of transformations to be used in sklearn pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'sklearn==0.21.3',
        'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True
)
