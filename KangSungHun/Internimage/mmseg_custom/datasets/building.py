from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BuildingDataset(CustomDataset):
    """Building dataset.
    """
    CLASSES = ('Building')

    PALETTE = [[165, 42, 42]]

    def __init__(self, **kwargs):
        super(BuildingDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)