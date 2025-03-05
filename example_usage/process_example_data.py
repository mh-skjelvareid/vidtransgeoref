import datetime
from pathlib import Path

from vidtransgeotag import VidTransGeoTag

# Paths
video_path = Path(
    r"/home/mha114/Dropbox/Python/vidtransgeotag/example_data/example_input/short_example_video.mp4"
)
csv_path = Path(
    r"/home/mha114/Dropbox/Python/vidtransgeotag/example_data/example_input/2025-03-01_143053_BodoVidTransTest.csv"
)
output_dir = Path("/media/mha114/Massimal2/tmp/test_vidtransgeotag")
gpkg_path = output_dir / "geotagged_images.gpkg"
image_dir = output_dir / "images"

# Create image directory
image_dir.mkdir(exist_ok=True)

# Create VidTransGeoTag object
vidtransgeotag = VidTransGeoTag(
    csv_file_path=csv_path,
    csv_time_add_offset=datetime.timedelta(hours=0),
    video_time_add_offset=datetime.timedelta(hours=-1),
)

# Check if video overlaps track
vidtransgeotag.check_video_overlaps_track(video_path=video_path, verbose=True)

# Extract geotagged images from video
vidtransgeotag.extract_geotagged_images_from_video(
    video_path, image_dir, gpkg_path=gpkg_path, filter_min_distance_m=1.0
)
