import numpy as np

from samit.inference.scene import scene_confusions


def test_overlap_blends_probabilities_and_counts_each_labeled_pixel_once():
    target = np.zeros((3, 4), dtype=np.uint8)
    target[0, 0] = 255
    left = np.broadcast_to(np.array([0.7, 0.3])[:, None, None], (2, 3, 3))
    right = np.broadcast_to(np.array([0.2, 0.8])[:, None, None], (2, 3, 3))

    tile, scene, counts = scene_confusions(
        target, [(0, 0, left), (0, 1, right)], 3, 2, 255
    )

    np.testing.assert_array_equal(tile, [[8, 9], [0, 0]])
    np.testing.assert_array_equal(scene, [[3, 8], [0, 0]])
    assert counts == {
        "tiles": 2,
        "labeled_pixels": 11,
        "covered_labeled_pixels": 11,
        "overlap_labeled_pixels": 6,
    }


def test_uncovered_labels_are_reported_and_tile_padding_is_cropped():
    target = np.array([[0, 1, 255, 0, 1]], dtype=np.uint8)
    probs = np.broadcast_to(np.array([1.0, 0.0])[:, None, None], (2, 3, 3))

    (tile, scene, counts), prediction, coverage = scene_confusions(
        target, [(0, 0, probs)], 3, 2, 255, include_prediction=True
    )
    np.testing.assert_array_equal(coverage, [[True, True, True, False, False]])
    np.testing.assert_array_equal(prediction, np.zeros_like(target))

    np.testing.assert_array_equal(tile, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(scene, tile)
    assert counts["labeled_pixels"] == 4
    assert counts["covered_labeled_pixels"] == 2
