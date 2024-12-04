"""Tests the VQA-Rad preprocessing."""

import logging
import pytest
from multimedeval.utils import BatcherInput, BatcherOutput
import numpy as np


logging.basicConfig(level=logging.INFO)


def dummy_list_arrays(n=1, dtype=np.uint8):
    """Returns a list of arrays."""
    return [np.zeros((10, 10), dtype=dtype) for _ in range(n)]


class TestDataclasses:
    """Tests the VQA-Rad preprocessing."""

    @pytest.mark.parametrize(
        "text, masks, should_success",
        [
            (None, None, False),
            (None, dummy_list_arrays(2), False),
            ("dummy text", None, True),
            ("dummy text", [], True),
            ("dummy text", dummy_list_arrays(2), False),
            ("<seg0> <seg1>", dummy_list_arrays(2), True),
            ("<seg0> <seg1>", dummy_list_arrays(3), False),
            ("<seg0> <seg1>", ["mask1", "mask2"], False),
            ("<seg0> <seg1>", dummy_list_arrays(1) + [None], False),
            ("<seg0> <seg1>", dummy_list_arrays(2, dtype=np.float32), False),
        ],
    )
    def test_batcher_output(self, text, masks, should_success):
        """Tests the batcher output dataclass."""

        if should_success:
            BatcherOutput(text=text, masks=masks)
        else:
            with pytest.raises(ValueError):
                BatcherOutput(text=text, masks=masks)

    # @pytest.mark.parametrize(
    #     "conversation, images, segmentation_masks, should_success",
    #     [
    #         (
    #             "are regions of the brain infarcted?",
    #             [],
    #             [],
    #             True,
    #         ),
    #     ],
    # )
    # def test_batcher_input(
    #     self, conversation, images, segmentation_masks, should_success
    # ):
    #     """Tests the batcher input dataclass."""

    #     if should_success:
    #         BatcherInput()
    #     else:
    #         with pytest.raises(ValueError):
    #             BatcherInput()
