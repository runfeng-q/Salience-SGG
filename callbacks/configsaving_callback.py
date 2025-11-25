from typing import Any

from lightning.pytorch.callbacks import Callback

from evaluation import calculate_mR_from_evaluator_list
from evaluation import CocoEvaluator, VGDetectEvaluator, BasicSceneGraphEvaluator, OICocoEvaluator, OIEvaluator, GQAEvaluator
from lightning.pytorch.utilities.types import STEP_OUTPUT


class ConfigSavingCallback(Callback):
    """A simple callback to log the config using the loggers of the trainer.
    """
    def __init__(self, evaluator, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator

    def on_test_start(self, trainer, pl_module):
        pl_module.eval()
        if self.evaluator=='VG_DETECT':
            detect_evaluator= VGDetectEvaluator(trainer.datamodule.test_dataset.coco, ['bbox'])
            evaluator={'detect_sgg': detect_evaluator}
        elif self.evaluator=='COCO':
            evaluator = CocoEvaluator(trainer.datamodule.test_dataset.coco, ['bbox'])
        elif self.evaluator=='VG_SGG':
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
            multiple_sgg_evaluator_list = []  # mR@k (for each rel category)
            single_sgg_evaluator_list = []
            if multiple_sgg_evaluator is not None:
                for index, name in enumerate(trainer.datamodule.test_dataset.rel_categories):
                    multiple_sgg_evaluator_list.append(
                        (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
                    )
            if single_sgg_evaluator is not None:
                for index, name in enumerate(trainer.datamodule.test_dataset.rel_categories):
                    single_sgg_evaluator_list.append(
                        (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=False))
                    )
            evaluator={'single_sgg_evaluator':single_sgg_evaluator,
                       'single_sgg_evaluator_list': single_sgg_evaluator_list,
            }
        elif self.evaluator=='GQA_SGG':
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
            multiple_sgg_evaluator_list = []  # mR@k (for each rel category)
            single_sgg_evaluator_list = []
            if multiple_sgg_evaluator is not None:
                for index, name in enumerate(trainer.datamodule.test_dataset.rel_categories):
                    multiple_sgg_evaluator_list.append(
                        (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
                    )
            if single_sgg_evaluator is not None:
                for index, name in enumerate(trainer.datamodule.test_dataset.rel_categories):
                    single_sgg_evaluator_list.append(
                        (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=False))
                    )
            evaluator = {'gqa_single_sgg_evaluator': single_sgg_evaluator,
                         'gqa_single_sgg_evaluator_list': single_sgg_evaluator_list,
                         }
        elif self.evaluator=='OI_DETECT':
            detect_evaluator= OICocoEvaluator(trainer.datamodule.test_dataset.rel_categories, trainer.datamodule.test_dataset.ind_to_classes, trainer.datamodule.test_dataset.targets)
            evaluator = {'oi_detect_sgg': detect_evaluator
                         }
        elif self.evaluator=='OI_SGG':
            oi_evaluator=OIEvaluator(trainer.datamodule.test_dataset.rel_categories,
                           trainer.datamodule.test_dataset.ind_to_classes)
            evaluator={'oi_sgg_evaluator' :oi_evaluator
                       }
        elif self.evaluator=='GQA_DETECT':
            gqa_evaluator=GQAEvaluator(trainer.datamodule.test_dataset.ind_to_classes)
            evaluator={'gqa_detect_sgg': gqa_evaluator}
        else:
            evaluator=None
        pl_module.evaluator = evaluator
        print('evaluator initialized')

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs!=None:
            self.save_examples(outputs, trainer, pl_module, batch)

    def on_test_end(self, trainer, pl_module):
        if pl_module.evaluator is not None:

            if 'single_sgg_evaluator' in pl_module.evaluator:
                recall = pl_module.evaluator['single_sgg_evaluator']["sgdet"].print_stats()
                mean_recall = calculate_mR_from_evaluator_list(
                    pl_module.evaluator['single_sgg_evaluator_list'], "sgdet", multiple_preds=False
                )
                for k, v in recall.items():
                    if k.startswith('z'):
                        continue
                    else:
                        print(f"f_{k}: {2/(1/mean_recall['m'+k]+1/v)}")
            if 'gqa_single_sgg_evaluator' in pl_module.evaluator:
                recall = pl_module.evaluator['gqa_single_sgg_evaluator']["sgdet"].print_stats()
                mean_recall = calculate_mR_from_evaluator_list(
                    pl_module.evaluator['gqa_single_sgg_evaluator_list'], "sgdet", multiple_preds=False
                )
                for k, v in recall.items():
                    if k.startswith('z'):
                        continue
                    else:
                        print(f"f_{k}: {2/(1/mean_recall['m'+k]+1/v)}")
            if 'detect_sgg' in pl_module.evaluator:
                pl_module.evaluator['detect_sgg'].synchronize_between_processes()
                pl_module.evaluator['detect_sgg'].accumulate()
                pl_module.evaluator['detect_sgg'].summarize()
            if 'sg_acc_evaluator' in pl_module.evaluator:
                pl_module.evaluator['sg_acc_evaluator'].summary()
            if 'oi_sgg_evaluator' in pl_module.evaluator:
                pl_module.evaluator['oi_sgg_evaluator'].aggregate_metrics()
            if 'oi_detect_sgg' in pl_module.evaluator:
                pl_module.evaluator['oi_detect_sgg'].aggregate_metrics()
            if 'gqa_detect_sgg' in pl_module.evaluator:
                pl_module.evaluator['gqa_detect_sgg'].aggregate_metrics()