from setuptools import setup,find_package

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ytdlpy',
    version='0.0.1',
    packages=find_package()
)
