from setuptools import setup,find_packages
install_requires = [
    # 必要な依存ライブラリがあれば記述
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ytdlpy',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=install_requires
)
