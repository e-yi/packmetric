from typing import Dict, Iterable, List, Union

import torch
from lightning.pytorch.utilities.memory import recursive_detach

from .metrics import BaseMetaMetric, MGMetric, BaseMetric
from .metrics import LEVEL_RUN, LEVEL_EPOCH, LEVEL_BATCH, STAGES

class MetricGroup(torch.nn.Module):
    """
    MetricGroup class integrates multiple metrics and meta-metrics into a single module.
    This class handles updating and aggregating metrics during various training stages.

    Attributes:
        input_metrics (List[MGMetric]): A list of metrics and meta-metrics.
        metrics (torch.nn.ModuleList): Stores instances of BaseMetric.
        meta_metrics (torch.nn.ModuleList): Stores instances of BaseMetaMetric.
    """

    def __init__(self, input_metrics: List[MGMetric]):
        super().__init__()

        metrics = []
        meta_metrics = []
        for metric in input_metrics:
            if isinstance(metric, BaseMetric):
                metrics.append(metric)
            elif isinstance(metric, BaseMetaMetric):
                meta_metrics.append(metric)
            else:
                raise ValueError(f'Metric type not understood: {type(metric)}')

        self.metrics = torch.nn.ModuleList(metrics)
        self.meta_metrics = torch.nn.ModuleList(meta_metrics)

    def add_metric(self, metric):
        """
        Add a metric to the appropriate group.

        Args:
            metric (BaseMetric or BaseMetaMetric): Metric to be added.
        """
        if isinstance(metric, BaseMetric):
            self.metrics.append(metric)
        elif isinstance(metric, BaseMetaMetric):
            self.meta_metrics.append(metric)
        else:
            raise ValueError(f'Metric type not understood: {type(metric)}')

    def __update_metrics(self, value: Dict, stage: str):
        for metric in self.metrics:
            metric.batch_update(**value, stage=stage)

    def __update_meta_metrics(self, metrics_value: Dict, stage: str):
        for meta_metric in self.meta_metrics:
            meta_metric.epoch_update(**metrics_value, stage=stage)

    def batch_step(self, step_output: Dict, stage: str):
        """
        Perform a batch update step by updating metrics and returning their values.

        Args:
            step_output (Dict): Outputs from a training/validation/test step.
            stage (str): Current stage of the process (train, val, test).

        Returns:
            Dict: Updated metric values for the current batch.
        """
        step_output = recursive_detach(step_output)

        self.__update_metrics(step_output, stage)

        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_value(stage=stage, level=LEVEL_BATCH))

        res = {f'{stage}/{k}': v for k, v in metrics.items()}

        return res

    def epoch_step(self, stage: str):
        """
        Aggregate and compute metric values at the end of an epoch.

        Args:
            stage (str): Current stage of the process (train, val, test).

        Returns:
            Dict: Aggregated metric values for the current epoch.
        """
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_value(stage=stage, level=LEVEL_EPOCH))

        self.__update_meta_metrics(metrics, stage)

        for meta_metric in self.meta_metrics:
            metrics.update(meta_metric.get_value(stage=stage))

        res = {f'{stage}/{k}': v for k, v in metrics.items()}

        return res

    def reset(self, level, stages: Union[str, Iterable[str]] = STAGES):
        """
        Reset metrics and meta-metrics based on the specified level and stage.

        Args:
            level (str): LEVEL_EPOCH or LEVEL_RUN specifying when to reset.
            stages (Union[str, Iterable[str]]): Stages during which reset should happen.
        """
        if level == LEVEL_BATCH:
            raise ValueError(f'Cannot reset: {level}')

        if isinstance(stages, str):
            stages = (stages,)

        for metric in self.metrics:
            for stage_i in stages:
                metric.reset(stage_i)

        if level == LEVEL_RUN:
            for meta_metric in self.meta_metrics:
                for stage_i in stages:
                    meta_metric.reset(stage_i)
