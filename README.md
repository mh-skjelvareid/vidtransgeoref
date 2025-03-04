# vidtransgeotag
Tools for generating geotagged ground truth images by merging video and
geolocation data streams.

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

## Quick start
Create a VidTransGeoTag object based on a CSV file with positions (latitude and
longitude), and set time offsets for CSV file and video files. Datetimes with timezone
information are converted to UTC, and datetimes without timezone information are assumed
to be given as UTC.

    vidtransgeotag = VidTransGeoTag(
        csv_path,
        csv_time_add_offset=datetime.timedelta(hours=0),
        video_time_add_offset=datetime.timedelta(hours=-1),
    )

In the example above, one hour is subtracted from the video timestamps to make them
align with the CSV timestamps. 

Check whether the track overlaps with a video file. Use "verbose" to print additional
information:

    vidtransgeotag.check_video_overlaps_track(
        video_path, 
        verbose=True)

Example output:

    short_example_video.mp4 starts at 2025-03-01 14:14:41+00:00 
    and ends at 2025-03-01 14:15:48.167100+00:00.
    Track starts at 2025-03-01 13:30:54+00:00 
    and ends at 2025-03-01 14:17:44+00:00
    Video is fully contained within track

It's possible to iterate on this, changing time offsets until the timing is correct. 

Extract images corresponding to every row of the CSV with timestamps overlapping the
video file. Save a GeoPackage file with the positions and filenames of each image.

    vidtransgeotag.extract_geotagged_images_from_video(
        video_path, 
        image_dir, 
        gpkg_path="geotagged_images.gpkg")

Optionally, filter points so that the change in position from point to point is above
a minimum distance (in meters):

    vidtransgeotag.extract_geotagged_images_from_video(
        video_path, 
        image_dir, 
        gpkg_path="geotagged_images.gpkg",
        filter_min_distance_m=10.0)


