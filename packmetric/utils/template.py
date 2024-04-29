from typing import Iterable

import torchmetrics

from packmetric.metrics import BaseMetaMetricAdapter, BaseMetricAdapter
from packmetric.metrics.base_metrics import STAGES


class MeanMetricAdapter(BaseMetricAdapter):
    """
    A specific implementation of BaseMetricAdapter using the MeanMetric from torchmetrics.
    This adapter calculates the mean of a metric across specified stages and logs the values
    at both batch and epoch levels.

    Attributes:
        name (str): The name of the metric, used for logging and identifying the metric in outputs.
        stages (Iterable[str]): Stages (e.g., 'train', 'val', 'test') during which the metric is active.
    """

    def __init__(self, name: str, stages: Iterable[str] = STAGES, ) -> None:
        super().__init__(name=name,
                         metric_init_fn=torchmetrics.MeanMetric,
                         input_pos_args=[name],
                         stages=stages,
                         metric_kwargs=None)


class MeanMetaMetricAdapter(BaseMetaMetricAdapter):
    """
    A specific implementation of BaseMetaMetricAdapter using the MeanMetric from torchmetrics.
    This adapter calculates the mean of meta-metric values across specified stages. It is designed
    to handle higher-level metric aggregations, providing insights at the epoch level.

    Attributes:
        name (str): The name of the meta-metric, used for logging and as part of the output dictionary keys.
        stages (Iterable[str]): Stages (e.g., 'train', 'val', 'test') during which the meta-metric is active.
    """

    def __init__(self, name: str, stages: Iterable[str] = STAGES) -> None:
        super().__init__(name=name,
                         metric_init_fn=torchmetrics.MeanMetric,
                         input_pos_args=[name],
                         stages=stages,
                         metric_kwargs=None)
