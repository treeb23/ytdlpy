from setuptools import setup,find_packages
install_requires = [
    ffmpeg,
    pydub,
    simpleaudio,
    youtube-transcript-api,
    yt-dlp,
    librosa,
    nltk,
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
