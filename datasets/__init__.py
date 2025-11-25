from .coco import COCODetectionDataModule
from .visual_genome import VGDataModule
from .open_image_v6 import OIDataModule
from .gqa import GQADataModule
DATA_SET_DICT={'COCO': COCODetectionDataModule, 'VG': VGDataModule,
               'OI': OIDataModule, 'GQA': GQADataModule
               }

def build_dataset(config):
    datamodule = DATA_SET_DICT[config['dataset'].split('_')[0]](config)
    return datamodule