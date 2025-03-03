from pathlib import Path

from vidtransgeotag import VidTransGeoTag

# Paths
# data_dir = Path("../example_data/example_input/2025-03-01_143053_BodoVidTransTest.csv")
video_path = Path(
    r"C:\Users\mha114\Dropbox\Python\vidtransgeotag\example_data\example_input\short_example_video.mp4"
)
csv_path = Path(
    r"C:\Users\mha114\Dropbox\Python\vidtransgeotag\example_data\example_input\2025-03-01_143053_BodoVidTransTest.csv"
)

print(csv_path.exists())

vidtransgeotag = VidTransGeoTag(csv_file_path=csv_path)

vidtransgeotag.check_video_overlaps_track(video_path=video_path, verbose=True)
