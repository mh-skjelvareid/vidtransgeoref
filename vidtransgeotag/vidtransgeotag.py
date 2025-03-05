# Imports
import datetime
import itertools
import os
import platform
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import dateutil.parser
import exiftool
import ffmpeg
import geopandas
import numpy as np
import pandas as pd
import pyproj
from numpy.typing import NDArray
from tqdm import tqdm

# Suppress "future" warnings (issue with shapely / geopandas)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Set exiftool path to exiftool.bat on Windows (installed via conda)
if platform.system().lower() == "windows":
    exiftool.constants.DEFAULT_EXECUTABLE = "exiftool.bat"  # type: ignore


# Note: Also check out batch geotegging of images
# https://help.propelleraero.com/hc/en-us/articles/19384091245719-How-to-Batch-Geotag-Photos-with-ExifTool


class VidTransGeoTag:
    def __init__(
        self,
        csv_file_path: Path,
        csv_time_add_offset: datetime.timedelta = datetime.timedelta(seconds=0),
        video_time_add_offset: datetime.timedelta = datetime.timedelta(seconds=0),
        csv_header_lat: Optional[str] = None,
        csv_header_lon: Optional[str] = None,
        csv_header_time: Optional[str] = None,
    ) -> None:
        self.csv_file_path = csv_file_path
        self.csv_time_add_offset = csv_time_add_offset
        self.video_time_add_offset = video_time_add_offset
        self.csv_header_lat = csv_header_lat
        self.csv_header_lon = csv_header_lon
        self.csv_header_time = csv_header_time

        self.track_gdf = self.read_track_csv()

    def read_track_csv(self):
        """Read GPS track from CSV file and convert to GeoDataFrame.

        This method reads a CSV file containing GPS track data and converts it into a GeoDataFrame
        with time and point geometry columns. It attempts to automatically detect column names
        for latitude, longitude and time data, or uses user-specified column names.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame with the following columns:
                - time : datetime
                    Timestamp for each point, with any specified offset added
                - geometry : Point
                    Point geometry containing longitude and latitude coordinates

        Raises
        ------
        ValueError
            If latitude, longitude or time columns cannot be found in the CSV file

        Notes
        -----
        The method looks for common column names for latitude (e.g., 'latitude', 'lat', 'y'),
        longitude (e.g., 'longitude', 'lon', 'x') and time (e.g., 'time', 'timestamp', 'datetime').
        Column names are case-insensitive.

        The resulting GeoDataFrame uses the WGS84 coordinate reference system (EPSG:4326).
        """
        # Read the CSV file
        df = pd.read_csv(self.csv_file_path)

        # Define plausible names for latitude, longitude, and time columns
        plausible_lat_names = ["latitude", "lat", "y", "Latitude", "Lat", "Y"]
        plausible_lon_names = [
            "longitude",
            "lon",
            "long",
            "lng",
            "x",
            "Longitude",
            "Lon",
            "Long",
            "Lng",
            "X",
        ]
        plausible_time_names = ["time", "timestamp", "datetime", "Time", "Timestamp", "Datetime"]

        # Determine the column names for latitude, longitude, and time
        lat_col = self.csv_header_lat or next(
            (col for col in plausible_lat_names if col in df.columns), None
        )
        lon_col = self.csv_header_lon or next(
            (col for col in plausible_lon_names if col in df.columns), None
        )
        time_col = self.csv_header_time or next(
            (col for col in plausible_time_names if col in df.columns), None
        )

        # Raise an error if any of the required columns are not found
        if not lat_col:
            raise ValueError("Latitude column not found in CSV file.")
        if not lon_col:
            raise ValueError("Longitude column not found in CSV file.")
        if not time_col:
            raise ValueError("Time column not found in CSV file.")

        # Convert to datetime
        df[time_col] = pd.to_datetime(df[time_col])
        # Handle both timezone-aware and naive datetimes
        df[time_col] = df[time_col].apply(
            lambda x: x.tz_convert("UTC") if x.tzinfo else x.tz_localize("UTC")
        )

        # Add offset if specified
        if self.csv_time_add_offset:
            df[time_col] = df[time_col] + self.csv_time_add_offset

        # Create a GeoDataFrame using latitude and longitude as geometry
        gdf = geopandas.GeoDataFrame(
            df[[time_col]],
            geometry=geopandas.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )  # type: ignore

        # Rename the time column
        gdf = gdf.rename(columns={time_col: "time"})

        return gdf

    def get_video_start_time_and_duration(
        self, video_path: Path
    ) -> tuple[datetime.datetime, datetime.timedelta]:
        """Get the creation time and duration of a video file.

        Parameters
        ----------
        video_path : Path
            Path object representing the location of the video file.

        Returns
        -------
        tuple[datetime.datetime, float]
            A tuple containing:
            - creation_time: The video creation time as datetime
            - duration: Duration of the video as timedelta

        Raises
        ------
        RuntimeError
            If there's an error probing the video file with ffmpeg.
        ValueError
            If the creation time or duration information is not found in the video metadata.

        """
        try:
            probe = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error probing video file: {e}")

        try:
            creation_time_str = probe["format"]["tags"]["creation_time"]
        except KeyError:
            raise ValueError("Creation time not found in video metadata")

        try:
            duration = float(probe["format"]["duration"])
        except KeyError:
            raise ValueError("Duration not found in video metadata")

        creation_time = self._normalize_datetime(dateutil.parser.parse(creation_time_str))
        creation_time += self.video_time_add_offset
        duration = datetime.timedelta(seconds=duration)
        return creation_time, duration

    def get_video_frame_rate(self, video_path: Path) -> float:
        """Get the frame rate of a video file.

        Parameters
        ----------
        video_path : Path
            Path object representing the location of the video file.

        Returns
        -------
        float
            The frame rate in frames per second.

        Raises
        ------
        RuntimeError
            If there's an error probing the video file with ffmpeg.
        ValueError
            If the frame rate information is not found in the video metadata.
        """
        try:
            probe = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error probing video file: {e}")

        for key in ["avg_frame_rate", "r_frame_rate"]:
            frame_rate_str = probe["streams"][0].get(key, "0/1")
            if frame_rate_str not in ("", "N/A"):
                try:
                    num, denom = map(int, frame_rate_str.split("/"))  # e.g. "30000/1001"
                    return num / denom
                except (ValueError, ZeroDivisionError):
                    continue  # Try the next key

        raise ValueError("Valid frame rate not found in video metadata")

    def filter_gdf_on_distance(
        self,
        gdf: geopandas.GeoDataFrame,
        epsg: Optional[int] = None,
        min_distance: float = 1.0,
        outlier_distance: float = 1000.0,
    ) -> geopandas.GeoDataFrame:
        """Filter a geodataframe by including samples only when position has changed significantly.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geodataframe to filter
        epsg : int, optional
            EPSG code for CRS to measure distance in. If None, best matching UTM zone will be used
        min_distance : float, default=1.0
            Minimum change in position required for next sample to be included.
            Units defined by CRS
        outlier_distance : float, default=1000.0
            Maximum allowed distance between consecutive points.
            Points with larger distances are excluded. Units defined by CRS

        Returns
        -------
        geopandas.GeoDataFrame
            Filtered geodataframe containing only points that meet the distance criteria

        Notes
        -----
        When epsg is None, the function automatically selects the UTM zone based on data centroid
        """
        # Find EPSG code for UTM zone if not provided
        if epsg is None:
            # Calculate centroid of all points
            centroid = gdf.geometry.unary_union.centroid
            # Use pyproj to find best UTM projection
            proj_string = pyproj.database.query_utm_crs_info(  # type: ignore
                datum_name="WGS 84",
                area_of_interest=pyproj.aoi.AreaOfInterest(  # type: ignore
                    west_lon_degree=centroid.x,
                    south_lat_degree=centroid.y,
                    east_lon_degree=centroid.x,
                    north_lat_degree=centroid.y,
                ),
            )
            epsg = proj_string[0].code

        # Convert to geomety suited to measure distance, e.g. UTM
        geom = gdf.geometry.to_crs(epsg=epsg)

        # Iterate over all positions, and only include a new point if position
        # has changed more than sample_distance
        mask = [0]  # Always include first point
        last_pos = geom.iloc[0]  # Position at first point
        for index, position in enumerate(geom):
            dist = position.distance(last_pos)
            if (dist > min_distance) and (dist < outlier_distance):
                mask.append(index)
                last_pos = position

        # Return a filtered copy of the original geodataframe
        return gdf.iloc[mask]

    def _normalize_datetime(self, dt: datetime.datetime) -> datetime.datetime:
        """Normalize datetime to UTC if timezone aware, or add UTC if naive.

        Parameters
        ----------
        dt : datetime.datetime or pandas.Timestamp
            The datetime object to normalize

        Returns
        -------
        datetime.datetime
            Normalized datetime object in UTC
        """
        if isinstance(dt, pd.Timestamp):
            if dt.tz is None:
                return dt.tz_localize("UTC")
            return dt.tz_convert("UTC")

        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)

    def check_video_overlaps_track(
        self,
        video_path: Path,
        verbose: bool = False,
    ) -> bool:
        """Check if video timestamps overlap with track timestamps.

        This function compares the start and end timestamps of a video file with the
        timestamps in the track data to determine if there is any temporal overlap.

        Parameters
        ----------
        video_path : Path
            Path to the video file to check
        verbose : bool, optional
            If True, prints detailed information about the temporal relationship
            between video and track, by default False

        Returns
        -------
        bool
            True if video timestamps overlap with track timestamps, False otherwise

        """
        video_start_time, video_duration = self.get_video_start_time_and_duration(video_path)
        video_end_time = video_start_time + video_duration

        track_start_time = self._normalize_datetime(self.track_gdf.time.min())
        track_end_time = self._normalize_datetime(self.track_gdf.time.max())

        if verbose:
            print(f"{video_path.name} starts at {video_start_time} and ends at {video_end_time}.")
            print(f"Track starts at {track_start_time} and ends at {track_end_time}")

        if video_end_time < track_start_time:
            if verbose:
                print("Video ends before track starts")
            return False
        elif video_start_time > track_end_time:
            if verbose:
                print("Video starts after track ends")
            return False
        elif (video_start_time > track_start_time) and (video_end_time < track_end_time):
            if verbose:
                print("Video is fully contained within track")
            return True
        else:
            if verbose:
                print("Video partly overlaps with track")
            return True

    def get_track_points_within_video(
        self,
        video_path: Path,
    ) -> geopandas.GeoDataFrame:
        """Identify track positions contained within video time window

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing only rows whose timestamps overlap with video time window
        """
        # Get video time window
        video_start_time, video_duration = self.get_video_start_time_and_duration(video_path)

        # Filter GeoDataFrame to only include rows within video time window
        mask = (self.track_gdf.time >= video_start_time) & (
            self.track_gdf.time <= (video_start_time + video_duration)
        )
        overlapping_gdf = geopandas.GeoDataFrame(self.track_gdf[mask], crs=self.track_gdf.crs)  # type: ignore

        return overlapping_gdf

    def images_from_video(
        self,
        video_input_file: Path,
        video_frame_rate: float,
        image_base_name: str,
        image_output_dir: Path,
        times: NDArray,
        batch_size: int = 20,
        image_quality: int = 5,
        overwrite: bool = True,
    ):
        """Extract multiple images from video at given times

        Parameters
        ----------
        video_input_file : str or Path
            Path to input video file
        image_base_name : str
            Base name for output image files (used during renaming)
        image_output_dir : Path
            Directory in which to output images
        times : list or array-like
            List of timestamps in seconds to extract frames from
        batch_size : int, optional
            Number of time points to process per ffmpeg call, by default 20
        image_quality : int, optional
            FFmpeg quality parameter (1-31), by default 5.
            Lower values result in higher quality images.
        overwrite : bool, optional
            Whether to overwrite existing files, by default True

        Returns
        -------
        list
            List of generated output filenames
        """
        IMAGE_OUTPUT_TEMPLATE = image_output_dir / "image_%d.jpg"
        SEEK_WINDOW_MARGIN = 1.0  # seconds

        # Sort times to ensure sequential access
        times = np.sort(times)

        # Calculate frame numbers for frames closest to each timestamp
        frames = (np.array(times) * video_frame_rate).astype(int)

        # Divide times and frames into batches
        n_batches = (len(times) // batch_size) + 1  # Add 1 to avoid possible zero
        times_batched = np.array_split(times, n_batches)
        frames_batched = np.array_split(frames, n_batches)

        # Initialize list to store image paths
        image_paths = []

        for time_batch, frame_batch in tqdm(zip(times_batched, frames_batched), total=n_batches):
            seek_start = time_batch[0] - SEEK_WINDOW_MARGIN
            seek_duration = time_batch[-1] - time_batch[0] + 2 * SEEK_WINDOW_MARGIN
            frame_offset = int(seek_start * video_frame_rate)

            # Create select_frames filter
            select_expr = "+".join(
                f"eq(n,{frame_number - frame_offset})" for frame_number in frame_batch
            )  # When using seek, frame numbering is relative to seek start

            # Call FFMPEG to extract images
            try:
                out, _ = (
                    # Use -ss and -t to restrict processing to the relevant segment.
                    ffmpeg.input(video_input_file, ss=seek_start, t=seek_duration)
                    .filter("select", select_expr)
                    .filter("settb", "AVTB")  # Fix timestamp basis
                    .output(
                        str(IMAGE_OUTPUT_TEMPLATE),
                        format="image2",
                        vsync="0",  # Prevent frame duplication
                        vcodec="mjpeg",
                        **{"q:v": image_quality},  # Set image quality
                    )
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=overwrite)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise

            # Generate list of created image files
            temp_image_files = [
                Path(str(IMAGE_OUTPUT_TEMPLATE).replace("%d", str(i + 1)))
                for i in range(len(frame_batch))
            ]

            # Rename image files with video name and timestamp
            image_paths.extend(
                self._rename_image_files_with_timestamp(
                    temp_image_files, image_base_name, time_batch
                )
            )

        return image_paths

    def _rename_image_files_with_timestamp(
        self, image_files: list[Path], image_base_name: str, image_times: NDArray
    ) -> list[Path]:
        """Rename image files, combining a base name with a time stamp.

        Parameters
        ----------
        image_files : list[Path]
            List of image file paths to be renamed.
        image_base_name : str
            Base name (typically the video file stem) to prepend to the timestamp.
        image_times : NDArray
            Array of time deltas (in seconds) relative to video start time.

        Returns
        -------
        list[Path]
            List of renamed image file paths.

        Raises
        ------
        FileNotFoundError
            If any specified image file does not exist.
        Exception
            If any error occurs during the renaming process.
        """
        new_image_files = []
        for image_file, image_time in zip(image_files, image_times):
            minutes = int(image_time // 60)
            seconds = int(image_time % 60)
            milliseconds = int((image_time % 1) * 1000)
            time_string = f"{minutes:03d}m{seconds:02d}s{milliseconds:03d}ms"
            new_image_file = (
                image_file.parent / f"{image_base_name}_{time_string}{image_file.suffix}"
            )
            try:
                shutil.move(image_file, new_image_file)
            except Exception as e:
                raise Exception(f"Failed to rename {image_file} to {new_image_file}: {e}")
            new_image_files.append(new_image_file)
        return new_image_files

    def extract_geotagged_images_from_video(
        self,
        video_path: Path,
        image_output_folder: Path,
        gpkg_path: Optional[Path] = None,
        filter_min_distance_m: Optional[float] = None,
    ) -> Optional[geopandas.GeoDataFrame]:
        """Extract geotagged images from video at timestamps overlapping with track

        Parameters
        ----------
        video_path : Path
            Path to input video file
        image_output_folder : Path
            Path to output folder where images will be saved

        Returns
        -------
        list
            List of generated output filenames
        """
        # Check if video overlaps with track
        if not self.check_video_overlaps_track(video_path):
            print("Video does not overlap with track. No images extracted.")
            return

        # Get timestamps overlapping with video duration
        track_gdf_within_video = self.get_track_points_within_video(video_path)
        if filter_min_distance_m:
            track_gdf_within_video = self.filter_gdf_on_distance(
                track_gdf_within_video, min_distance=filter_min_distance_m
            )
        track_timestamps = track_gdf_within_video.time

        # Calculate image times (in seconds) relative to video start time
        video_start_time, _ = self.get_video_start_time_and_duration(video_path)
        video_frame_rate = self.get_video_frame_rate(video_path)
        image_time_relative_to_video_start = pd.TimedeltaIndex(
            track_timestamps - video_start_time
        ).total_seconds()

        # Extract images at overlapping timestamps
        image_files = self.images_from_video(
            video_path,
            video_frame_rate,
            image_base_name=video_path.stem,
            image_output_dir=image_output_folder,
            times=image_time_relative_to_video_start,  # type: ignore
        )

        # Add image filenames to track_gdf_within_video
        track_gdf_within_video["image_file"] = [f.name for f in image_files]

        # Save to GeoPackage if path is provided
        if gpkg_path:
            track_gdf_within_video.to_file(gpkg_path, driver="GPKG")

        return track_gdf_within_video

    def batch_extract_geotagged_images_from_videos(
        self, video_folder: Path, image_output_folder: Path, gpkg_path: Optional[Path] = None
    ) -> geopandas.GeoDataFrame:
        """Process multiple videos to extract and geotag frames at track locations.

        This method processes all MP4 video files in the specified folder, extracting frames
        at timestamps that match GPS track points and embedding the corresponding location
        data in each image's EXIF metadata.

        Parameters
        ----------
        video_folder : Path
            Path to folder containing MP4 video files to process
        image_output_folder : Path
            Path to folder where extracted and geotagged images will be saved.
            Images are named using pattern: videoname_MMMmSSsMMMs.jpg where:
            - MMM = minutes (zero-padded to 3 digits)
            - SS = seconds (zero-padded to 2 digits)
            - MMM = milliseconds (zero-padded to 3 digits)
        gpkg_path : Optional[Path], default=None
            If provided, saves the combined GeoDataFrame to this path as a GeoPackage file

        Returns
        -------
        geopandas.GeoDataFrame
            Combined GeoDataFrame containing extracted image locations and metadata with columns:
            - geometry : Point
                Location where image was taken (from GPS track)
            - time : datetime
                Timestamp when image was taken
            - image_file : Path
                Path to the extracted and geotagged image file

        Notes
        -----
        - Only processes .mp4 files in the video folder
        - Only extracts frames from videos that temporally overlap with the GPS track
        - Returns empty GeoDataFrame if no valid data is found
        - Uses EPSG:4326 (WGS84) coordinate reference system
        """
        video_files = list(video_folder.glob("*.[mM][pP]4"))
        gdfs_with_image_file_names = []

        for video_file in tqdm(video_files, desc="Processing videos"):
            gdf = self.extract_geotagged_images_from_video(video_file, image_output_folder)
            if isinstance(gdf, geopandas.GeoDataFrame):  # Only append valid GeoDataFrames
                gdfs_with_image_file_names.append(gdf)

        if not gdfs_with_image_file_names:
            print("No valid data found. Returning empty GeoDataFrame.")
            return geopandas.GeoDataFrame(geometry=[], crs="EPSG:4326")  # type: ignore # Return empty GeoDataFrame if no valid data

        # Combine all GeoDataFrames into a single one using GeoDataFrame.concat()
        combined_gdf = geopandas.GeoDataFrame(
            pd.concat(gdfs_with_image_file_names, ignore_index=True),
            crs=gdfs_with_image_file_names[0].crs,
        )  # type: ignore

        # Save to GeoPackage if path is provided
        if gpkg_path:
            combined_gdf.to_file(gpkg_path, driver="GPKG")

        return combined_gdf


def write_geotag_to_image(
    image_path: Path,
    lat: float,
    lon: float,
    altitude: Optional[float] = None,
    timestamp: Optional[datetime.datetime] = None,
) -> None:
    """Write GPS coordinates to image EXIF metadata.

    Parameters
    ----------
    image_path : Path
        Path to the image file
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    altitude : float, optional
        Altitude in meters
    timestamp : datetime.datetime, optional
        GPS timestamp to write

    Notes
    -----
    Uses exiftool to write the following GPS tags:
    - GPSLatitude/GPSLatitudeRef
    - GPSLongitude/GPSLongitudeRef
    - GPSAltitude/GPSAltitudeRef (if altitude provided)
    - GPSDateStamp/GPSTimeStamp (if timestamp provided)
    """
    with exiftool.ExifToolHelper() as et:
        params = {
            "GPSLatitude": lat,
            "GPSLatitudeRef": "N" if lat >= 0 else "S",
            "GPSLongitude": lon,
            "GPSLongitudeRef": "E" if lon >= 0 else "W",
        }

        if altitude is not None:
            params.update(
                {"GPSAltitude": abs(altitude), "GPSAltitudeRef": 0 if altitude >= 0 else 1}
            )

        if timestamp is not None:
            params.update(
                {
                    "GPSDateStamp": timestamp.strftime("%Y:%m:%d"),
                    "GPSTimeStamp": timestamp.strftime("%H:%M:%S"),
                }
            )

        et.set_tags(str(image_path), params)


def merge_videos_in_directory(
    input_dir: Path,
    output_file: Path,
    compress_by_transcoding: bool = False,
    transcoding_quality_crf=28,
) -> None:
    """
    Merges all MP4 video files in the given directory into one file without transcoding.
    Assumes the files (sorted alphabetically) are from a single recording. The creation_time
    metadata from the first file is transferred to the merged output.

    Parameters
    ----------
    input_dir : Path
        Directory containing the MP4 video files.
    output_file : Path
        Path to the merged output video file.
    compress_by_transcoding : bool, optional
        If True, the videos are compressed by transcoding to H.264 using libx264, by default False
    transcoding_quality_crf : int, optional
        Constant Rate Factor (CRF) for libx264 (0-51, lower is higher quality), by
        default 28. Ignored if compress_by_transcoding is False.


    Raises
    ------
    ValueError
        If no MP4 files are found in the provided directory.
    RuntimeError
        If probing the first file fails or ffmpeg fails to merge the videos.

    Notes
    -----
    - The constant rate factor (CRF) controls the quality of the output video. A value
      of 23 is often considered a good compromise between quality and file size. A default
      of 28 is used here for slightly more aggressive compression.
    """
    # List and sort all MP4 files in the directory
    video_files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"]
    )
    if not video_files:
        raise ValueError("No MP4 video files found in the provided directory.")

    # Probe the first video file to extract the creation time metadata
    try:
        probe = ffmpeg.probe(str(video_files[0]))
        creation_time = probe.get("format", {}).get("tags", {}).get("creation_time")
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error probing the first video file: {e}")

    # Create a temporary file list for the ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as file_list:
        for video in video_files:
            # Using absolute paths is recommended when using -safe 0
            file_list.write(f"file '{video.resolve()}'\n")
        list_file_path = file_list.name

    try:
        # Create an input stream from the file list using the concat demuxer.
        # The 'safe' flag is set to 0 to allow absolute paths.
        input_stream = ffmpeg.input(list_file_path, format="concat", safe=0)

        # Build output parameters.
        if compress_by_transcoding:
            output_params = {
                "vcodec": "libx264",
                "crf": transcoding_quality_crf,
            }
        else:
            output_params = {"c": "copy"}  # Copy mode, no transcoding

        if creation_time:
            output_params["metadata"] = f"creation_time={creation_time}"

        # Build and run the ffmpeg command using the ffmpeg-python chain.
        stream = ffmpeg.output(input_stream, str(output_file), **output_params)
        ffmpeg.run(stream, overwrite_output=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed to merge videos: {e}")
    finally:
        # Clean up the temporary file list.
        os.remove(list_file_path)
