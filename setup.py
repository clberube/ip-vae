# @Author: charles
# @Date:   2021-07-27 09:07:80
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-07-27 09:07:65


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IP-VAE",
    version="0.0.1",
    author="Charles L. Bérubé",
    author_email="charles.berube@polymtl.ca",
    description="Data-driven IP modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/clberube/ip-vae/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "ipvae"},
    packages=setuptools.find_packages(where="ipvae"),
    python_requires=">=3.6",
)
