import datetime
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest

from vidtransgeotag import VidTransGeoTag


@pytest.fixture
def example_data_paths():
    base_path = Path(__file__).parent.parent / "example_data" / "example_input"
    video_path = base_path / "short_example_video.mp4"
    csv_path = base_path / "2025-03-01_143053_BodoVidTransTest.csv"
    return video_path, csv_path


@pytest.fixture
def expected_output_paths():
    base_path = Path(__file__).parent.parent / "example_data" / "example_output"
    expected_image_dir = base_path / "images"
    expected_gpkg_path = base_path / "geotagged_images.gpkg"
    return expected_image_dir, expected_gpkg_path


def test_check_video_overlaps_track(example_data_paths):
    video_path, csv_path = example_data_paths

    vidtransgeotag = VidTransGeoTag(
        csv_file_path=csv_path,
        csv_time_add_offset=datetime.timedelta(hours=0),
        video_time_add_offset=datetime.timedelta(hours=-1),
    )

    assert vidtransgeotag.check_video_overlaps_track(video_path=video_path)


def test_extract_geotagged_images_from_video(example_data_paths, expected_output_paths):
    video_path, csv_path = example_data_paths
    expected_image_dir, expected_gpkg_path = expected_output_paths

    with tempfile.TemporaryDirectory() as temp_output_dir:
        output_dir = Path(temp_output_dir)
        image_dir = output_dir / "images"
        gpkg_path = output_dir / "geotagged_images.gpkg"

        image_dir.mkdir(exist_ok=True)

        vidtransgeotag = VidTransGeoTag(
            csv_file_path=csv_path,
            csv_time_add_offset=datetime.timedelta(hours=0),
            video_time_add_offset=datetime.timedelta(hours=-1),
        )

        track_gdf_within_video = vidtransgeotag.extract_geotagged_images_from_video(
            video_path, image_dir, gpkg_path=gpkg_path, filter_min_distance_m=1.0
        )

        assert track_gdf_within_video is not None

        # Compare generated images with expected images
        generated_images = sorted(image_dir.glob("*.jpg"))
        expected_images = sorted(expected_image_dir.glob("*.jpg"))
        assert len(generated_images) == len(expected_images)

        for generated_image, expected_image in zip(generated_images, expected_images):
            assert generated_image.name == expected_image.name

        # Check properties of generated geopackage
        assert gpkg_path.exists()
        generated_gdf = gpd.read_file(gpkg_path)
        expected_gdf = gpd.read_file(expected_gpkg_path)
        assert list(generated_gdf.columns) == list(expected_gdf.columns)
        assert len(generated_gdf) == len(expected_gdf)


if __name__ == "__main__":
    pytest.main()
