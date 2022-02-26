from setuptools import setup

setup(
    name='MSAI_495_Project',
    version='1.0',
    description='A useful module',
    author="Dimitrios Mavrofridis, Aleksandr Simonyan, Donald Baracskay, Josh Cheema",
    author_email='dimitriosmavrofridis@gmail.com',
    packages=['MSAI_495_Project'],  # same as name
    install_requires=["gdown", "pandas", "mat73", "matplotlib", "seaborn", "torch", "opencv-python"]
)
