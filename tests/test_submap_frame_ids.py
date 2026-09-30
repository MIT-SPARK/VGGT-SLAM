import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vggt_slam.submap import Submap


def test_set_frame_ids_uses_trailing_number_not_prefix_digit():
    # Filenames where a digit appears earlier in the name (a camera index
    # or session prefix) in addition to the real trailing frame number.
    submap = Submap(0)
    submap.set_frame_ids([
        "cam0_frame_012.png",
        "img2_0015.png",
        "frame_0001.jpg",
    ])

    assert submap.get_frame_ids() == [12.0, 15.0, 1.0]
