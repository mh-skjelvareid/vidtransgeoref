import datetime
from pathlib import Path

from exiftool import constants

from vidtransgeotag import VidTransGeoTag

# Set exiftool default executable to .bat version (installed via conda)
# constants.DEFAULT_EXECUTABLE = "exiftool.bat"

# Paths
# data_dir = Path("../example_data/example_input/2025-03-01_143053_BodoVidTransTest.csv")
# video_path = Path(
#     r"C:\Users\mha114\Dropbox\Python\vidtransgeotag\example_data\example_input\short_example_video.mp4"
# )
# csv_path = Path(
#     r"C:\Users\mha114\Dropbox\Python\vidtransgeotag\example_data\example_input\2025-03-01_143053_BodoVidTransTest.csv"
# )

video_path = Path(
    r"/home/mha114/Dropbox/Python/vidtransgeotag/example_data/example_input/short_example_video.mp4"
)
csv_path = Path(
    r"/home/mha114/Dropbox/Python/vidtransgeotag/example_data/example_input/2025-03-01_143053_BodoVidTransTest.csv"
)
output_dir = Path("/home/mha114/Dropbox/Python/vidtransgeotag/example_data/example_output")
image_dir = output_dir / "images"
gpkg_path = output_dir / "geotagged_images.gpkg"

vidtransgeotag = VidTransGeoTag(
    csv_file_path=csv_path,
    csv_time_add_offset=datetime.timedelta(hours=0),
    video_time_add_offset=datetime.timedelta(hours=-1),
)

vidtransgeotag.check_video_overlaps_track(video_path=video_path, verbose=True)

# vidtransgeotag.extract_geotagged_images_from_video(video_path, image_dir, gpkg_path=gpkg_path)
