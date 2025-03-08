[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14974704.svg)](https://doi.org/10.5281/zenodo.14974704)

# vidtransgeotag
Tools for generating geotagged ground truth images by merging video and geolocation data
streams.

# Documentation
[VidTransGeoTag documentation](https://mh-skjelvareid.github.io/vidtransgeotag/).

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

Note that the geotagging information is currently only saved in the GeoPackage file, and
not written as EXIF metadata in the image files. However, writing EXIF metadata is a
goal for future work on the package. 

## Merging video split across multiple files
vidtransgeotag uses ffmpeg to read metadata from video files and find the timestamp for
the start of the recording. Note that cameras that split long recordings across multiple
files (e.g. GoPro) may use the same timestamp for all files, even though it's only
really valid for the first file. This can cause problems when coordinating the
timestamps in the position log and in the video files. 

To avoid this problem, vidtransgeotag provides a function for merging multiple video
files into one, while retaining the timestamp in the metadata. All files from a single
recording are assumed to be placed inside the same directory (without other video
files), and the alphabetical ordering of file names is assumed to match the
chronological ordering of the video files. Example:

    vidtransgeotag.merge_videos_in_directory(
        input_video_dir, 
        merged_video_path)

By default the videos are merged without transcoding, which is fast and lossless.
However, the merged file can be quite big. It is possible to transcode and compress the
video while merging, although this will be much slower:

    vidtransgeotag.merge_videos_in_directory(
        input_video_dir, 
        merged_video_path,
        compress_by_transcoding=True)


