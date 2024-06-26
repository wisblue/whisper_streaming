# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib

from setuptools import setup, find_packages
import setuptools.command.build_py


here = pathlib.Path(__file__).parent.resolve()
about = {}
with open(os.path.join(here, "version.py"), "r") as f:
    exec(f.read(), about)


setup(
    name="whisper_streaming",
    version=about["__version__"],
    description="Whisper realtime streaming for long speech-to-text transcription and translation.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ufal/whisper_streaming",
    author=" Dominik Macháček, Raj Dabre, Ondřej Bojar",
    cmdclass={},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=["whisper", "realtime", "audio", "transcrie", "streaming"],
    license="MIT",
    packages=find_packages(),
    include_package_data=True,  # Include data files (optional)
    python_requires=">=3.9.0",
    install_requires=[
        "librosa",
        "soundfile",
        "faster-whisper",
    ],
    package_data={},
    project_urls={
        "Documentation": "https://github.com/ufal/whisper_streaming/blob/main/README.md",
        "Website": "https://livekit.io/",
        "Source": "https://github.com/ufal/whisper_streaming",
    },
)
