# vidtransgeotag
Tools for generating geotagged ground truth images by merging video and
geolocation data streams.

UNDER DEVELOPMENT - NOT READY FOR END USERS

## Dependencies
The module requires the binaries FFMPEG and ExifTool to be installed on your system. The
Python GeoPandas package also requires non-Python libraries. One simple way to install
these is via Conda:

    conda create -n vidtransgeotag -c conda-forge python=3.12 ffmpeg exiftool geopandas

Activate the environment afterwards:

    conda activate vidtransgeotag

## Installation
Download the source code, navigate to the root of the source code folder, and install
the package with

    pip install .

For an editable install with additional developer tools, use

    pip install -e .[dev]
