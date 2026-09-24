from setuptools import setup

setup(
    name="cams_generation",
    py_modules=["cams_generation"],
    install_requires=["blobfile", "torch", "tqdm"],
    #install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
