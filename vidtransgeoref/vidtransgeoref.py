# Imports
import datetime
import re
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

import exiftool
import ffmpeg
import geopandas
import numpy as np
import pandas as pd
import pyproj
from dateutil import parser
from tqdm import tqdm

# Suppress "future" warnings (issue with shapely / geopandas)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Note: Also check out batch geotegging of images
# https://help.propelleraero.com/hc/en-us/articles/19384091245719-How-to-Batch-Geotag-Photos-with-ExifTool


class VidTransGeoTag:
    def __init__(
        self,
        csv_file_path: Path,
        csv_time_add_offset: datetime.timedelta = datetime.timedelta(seconds=0),
        csv_header_lat: Optional[str] = None,
        csv_header_lon: Optional[str] = None,
        csv_header_time: Optional[str] = None,
    ) -> None:
        self.csv_file_path = csv_file_path
        self.csv_time_add_offset = csv_time_add_offset
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
        plausible_lon_names = ["longitude", "lon", "x", "Longitude", "Lon", "X"]
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

        # Convert time column to datetime
        df[time_col] = pd.to_datetime(df[time_col])

        # Add offset if specified
        if self.csv_time_add_offset:
            df[time_col] = df[time_col] + self.csv_time_add_offset

        # Create a GeoDataFrame using latitude and longitude as geometry
        gdf = geopandas.GeoDataFrame(
            df[time_col],
            geometry=geopandas.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )

        # Rename the time column
        gdf = gdf.rename(columns={time_col: "time"})

        return gdf

    def get_video_creation_time(self, video_path: Path) -> datetime.datetime:
        """Extract video creation time from metadata.

        This function uses ExifTool to extract the video creation timestamp from various metadata
        fields, checking multiple tags in order of reliability.

        Parameters
        ----------
        video_path : Path
            Path object pointing to the video file

        Returns
        -------
        datetime.datetime
            The creation time of the video as a datetime object

        Raises
        ------
        ValueError
            If the creation time cannot be found in metadata or if the date format is unrecognized

        Notes
        -----
        The function checks the following metadata tags in order:
            1. QuickTime:CreateDate
            2. EXIF:DateTimeOriginal
            3. QuickTime:MediaCreateDate
            4. File:FileCreateDate
        """
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata(str(video_path))

            # Try multiple metadata tags in order of reliability
            creation_time_str = (
                metadata.get("QuickTime:CreateDate")
                or metadata.get("EXIF:DateTimeOriginal")
                or metadata.get("QuickTime:MediaCreateDate")
                or metadata.get("File:FileCreateDate")
            )

            if creation_time_str:
                try:
                    return parser.parse(creation_time_str)
                except parser.ParserError:
                    raise ValueError(f"Unrecognized date format: {creation_time_str}")
            else:
                raise ValueError("Creation time not found in video metadata")

    def get_video_duration(self, video_path: Path) -> float:
        """Get the duration of a video file in seconds.

        Parameters
        ----------
        video_path : Path
            Path object representing the location of the video file.

        Returns
        -------
        float
            Duration of the video in seconds.

        Raises
        ------
        RuntimeError
            If there's an error probing the video file with ffmpeg.
        ValueError
            If the duration information is not found in the video metadata.

        """
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe["format"]["duration"])
            return duration
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error probing video file: {e}")
        except KeyError:
            raise ValueError("Duration not found in video metadata")

    def filter_gdf_on_distance(
        self,
        gdf: geopandas.GeoDataFrame,
        epsg: Optional[int] = None,
        sample_distance: float = 1.0,
        outlier_distance: float = 1000.0,
    ) -> geopandas.GeoDataFrame:
        """Filter a geodataframe by including samples only when position has changed significantly.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geodataframe to filter
        epsg : int, optional
            EPSG code for CRS to measure distance in. If None, best matching UTM zone will be used
        sample_distance : float, default=1.0
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
            proj_string = pyproj.database.query_utm_crs_info(
                datum_name="WGS 84",
                area_of_interest=pyproj.aoi.AreaOfInterest(
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
            if (dist > sample_distance) and (dist < outlier_distance):
                mask.append(index)
                last_pos = position

        # Return a filtered copy of the original geodataframe
        return gdf.iloc[mask]

    def check_video_overlaps_track(self, video_path: Path, verbose: bool = False) -> bool:
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
        video_start_time = self.get_video_creation_time(video_path)
        video_end_time = video_start_time + datetime.timedelta(
            seconds=self.get_video_duration(video_path)
        )
        track_start_time = self.track_gdf["time"].min()
        track_end_time = self.track_gdf["time"].max()

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

    def get_track_points_within_video(self, video_path: Path) -> geopandas.GeoDataFrame:
        """Identify track positions contained within video time window

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing only rows whose timestamps overlap with video time window
        """
        # Get video time window
        video_start_time = self.get_video_creation_time(video_path)
        video_duration = datetime.timedelta(seconds=self.get_video_duration(video_path))

        # Filter GeoDataFrame to only include rows within video time window
        mask = (self.track_gdf["time"] >= video_start_time) & (
            self.track_gdf["time"] <= (video_start_time + video_duration)
        )
        overlapping_gdf = self.track_gdf[mask]

        return overlapping_gdf

    def images_from_video(
        self,
        video_input_file,
        times,
        image_output_template: Union[str, Path] = "image_%06d.jpg",
        image_quality=5,
        overwrite=True,
    ):
        """Extract multiple images from video at given times

        Parameters
        ----------
        video_input_file : str or Path
            Path to input video file
        image_output_template : str or Path
            Template for output image files. Must contain '%0<n_digits>d' which will be
            replaced with the frame number. Example: 'image_%06d.jpg' (zero-pads to 6
            digits)
        times : list or array-like
            List of timestamps in seconds to extract frames from
        image_quality : int, optional
            FFmpeg quality parameter (1-31), by default 5
        overwrite : bool, optional
            Whether to overwrite existing files, by default True

        Returns
        -------
        list
            List of generated output filenames

        Raises
        ------
        ValueError
            If image_output_template doesn't contain proper format specifier
        """
        # Extract the format specifier pattern for template filename (e.g., '%06d')
        image_numbering_spec = re.search(r"%0\d+d", str(image_output_template))
        if not image_numbering_spec:
            raise ValueError("image_output_template must contain format specifier like '%06d'")
        image_numbering_spec = image_numbering_spec.group(0)  # Get full re match (e.g. '%06d')

        # Sort times to ensure sequential access
        times = sorted(times)

        # Create select_frames filter using 'nearest' mode
        select_expr = "+".join([f"not(mod(t,{t}))" for t in times])

        try:
            out, _ = (
                ffmpeg.input(video_input_file)
                .filter("select", select_expr)
                .filter("settb", "AVTB")  # Fix timestamp basis
                .output(
                    str(image_output_template),
                    format="image2",
                    vcodec="mjpeg",
                    **{"q:v": image_quality},
                )
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=overwrite)
            )

            # Get the number of digits in image file name from the format specifier
            image_ndigits = int(image_numbering_spec[2:-1])

            # Generate list of created files
            output_files = [
                Path(
                    str(image_output_template).replace(
                        image_numbering_spec, str(i + 1).zfill(image_ndigits)
                    )
                )
                for i in range(len(times))
            ]
            return output_files

        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise

    def write_gps_to_image(
        self,
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

    def rename_image_files_with_timestamp(
        self, image_files: list[Path], video_name: str, image_times: pd.Series[datetime.timedelta]
    ):
        """Rename image files with video name and timestamp.

        This function renames a list of image files using the video name and
        corresponding timestamps. The new filename format is:
        {video_name}_{minutes}m{seconds}s{milliseconds}ms.{original_extension}

        Parameters
        ----------
        image_files : list[Path]
            List of Path objects pointing to image files to be renamed
        video_name : str
            Name of the video file to use in the new filenames
        image_times : pd.Series[datetime.timedelta]
            Series of timedelta objects containing timestamps for each image

        Returns
        -------
        None

        Notes
        -----
        - Original files are moved/renamed in place
        - Timestamps are formatted as MMMmSSsMMMs where:
            - MMM = minutes (zero-padded to 3 digits)
            - SS = seconds (zero-padded to 2 digits)
            - MMM = milliseconds (zero-padded to 3 digits)
        """
        new_image_files = []
        for image_file, image_time in zip(image_files, image_times):
            total_seconds = image_time.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds % 1) * 1000)

            time_string = f"{minutes:03d}m{seconds:02d}s{milliseconds:03d}ms"
            new_image_file = image_file.parent / f"{video_name}_{time_string}.{image_file.suffix}"
            shutil.move(image_file, new_image_file)
            new_image_files.append(new_image_file)
        return new_image_files

    def extract_geotagged_images_from_video(self, video_path: Path, image_output_folder: Path):
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
        track_timestamps = track_gdf_within_video["time"]

        # Calculate image times (in seconds) relative to video start time
        video_start_time = self.get_video_creation_time(video_path)
        image_time_relative_to_video_start = track_timestamps - video_start_time

        # Extract images at overlapping timestamps
        image_output_template = image_output_folder / "image_%06d.jpg"
        image_files = self.images_from_video(
            video_path,
            times=image_time_relative_to_video_start.dt.total_seconds(),
            image_output_template=image_output_template,
        )

        # Write GPS data to each image
        for image_file, (_, row) in zip(image_files, track_gdf_within_video.iterrows()):
            self.write_gps_to_image(
                Path(image_file), lat=row.geometry.y, lon=row.geometry.x, timestamp=row.time
            )

        # Rename image files with video name and timestamp
        renamed_image_files = self.rename_image_files_with_timestamp(
            image_files, video_path.stem, image_time_relative_to_video_start
        )

        # Add image filenames to track_gdf_within_video
        track_gdf_within_video["image_file"] = renamed_image_files

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
            Combined GeoDataFrame containing all extracted image locations and metadata with columns:
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
            return geopandas.GeoDataFrame(
                geometry=[], crs="EPSG:4326"
            )  # Return empty GeoDataFrame if no valid data

        # Combine all GeoDataFrames into a single one using GeoDataFrame.concat()
        combined_gdf = geopandas.GeoDataFrame(
            pd.concat(gdfs_with_image_file_names, ignore_index=True),
            crs=gdfs_with_image_file_names[0].crs,
        )

        # Save to GeoPackage if path is provided
        if gpkg_path:
            combined_gdf.to_file(gpkg_path, driver="GPKG")

        return combined_gdf
