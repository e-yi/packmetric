import abc
from typing import Dict, Iterable, List, Optional

import torch

# Constants for different stages and levels of metric calculations.

STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
STAGE_TEST = 'test'

STAGES = (STAGE_TRAIN, STAGE_VAL, STAGE_TEST)

LEVEL_BATCH = 'batch'
LEVEL_EPOCH = 'epoch'
LEVEL_RUN = 'run'

LEVELS = (LEVEL_BATCH, LEVEL_EPOCH, LEVEL_RUN)


class MGMetric(abc.ABC, torch.nn.Module):
    """
    Abstract base class for all metric types in the framework, providing an interface
    for metrics and meta-metrics. All custom metrics should inherit from this class.
    """
    STAGE_PREFIX = 'stage_'
    pass


class BaseMetric(MGMetric):
    """
    Base class for metrics that are updated every batch and reset at the end of each epoch.
    This class processes outputs from methods like XXX_step() of a LightningModule, and it logs
    the metric values at specified intervals (either each batch or each epoch depending on the
    log levels set).

    It is designed to be subclassed to implement complicated metric calculations.
    For typical use cases, consider using `BaseMetricAdapter`.

    Pattern:
        - Updates: Each batch.
        - Inputs: Uses outputs from XXX_step() methods as inputs.
        - Logging: Logs metrics at each batch and each epoch based on configuration.
        - Resets: Resets metrics at the end of each epoch.

    Attributes:
        name (str): Name of the metric, which will be used during logging as `<stage>/<name>_<log_level>`.
        input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
        stages (Iterable[str]): The stages (e.g., 'train', 'val', 'test') at which the metric is active.
    """

    def __init__(self, name: str, input_pos_args: List[str], stages: Iterable[str]) -> None:
        """
        Args:
            name (str): Name of the metric, which will be used during logging as `<stage>/<name>_<log_level>`.
            input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
            stages (List[str]): The stages (e.g., 'train', 'val', 'test') at which the metric is active.
        """
        super().__init__()
        self.name = name
        self.input_pos_args = input_pos_args
        self.stages = stages

    @abc.abstractmethod
    def batch_update(self, stage, **kwargs):
        """
        Update the metric for a given batch.

        Args:
            stage (str): Current stage of training/validation/testing.
            **kwargs: Arbitrary keyword arguments containing input data needed for metric calculation.
        """
        pass

    @abc.abstractmethod
    def get_value(self, stage, level) -> Dict:
        """
        Retrieve the current value of the metric.

        Args:
            stage (str): Current stage.
            level (str): The level at which metrics are aggregated (batch, epoch, or run).

        Returns:
            Dict: Metric value dictionary formatted as {metric_name: value}.
        """
        pass

    @abc.abstractmethod
    def reset(self, stage):
        """
        Reset the metric calculations for a new epoch or run.

        Args:
            stage (str): Stage at which the reset should occur.
        """
        pass


class BaseMetricAdapter(BaseMetric):
    """
    Adapter for BaseMetric that facilitates the initialization and dynamic configuration of metrics.
    This class extends BaseMetric to support the creation of metric instances on-the-fly, based on the provided
    initialization function and additional metric-specific arguments.

    Pattern:
        - Updates: Each batch.
        - Inputs: Uses outputs from XXX_step() methods as inputs.
        - Logging: Logs metrics at each batch and/or each epoch based on configuration.
        - Resets: Resets metrics at the end of each epoch.

    Attributes:
        name (str): Name of the metric, which will be used during logging as `<stage>/<name>_<log_level>`.
        input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
        stages (Iterable[str],): The stages (e.g., 'train', 'val', 'test') at which the metric is active.
        metrics (torch.nn.ModuleDict): Contains instantiated metrics for each stage.
        input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
        metric_kwargs (Dict, optional): Additional keyword arguments to pass to metric_fn during `batch_update`.
    """

    def __init__(self, name: str, metric_init_fn,
                 input_pos_args: List[str], stages: Iterable[str] = STAGES,
                 metric_kwargs: Dict = None, fill_nan=True) -> None:
        """
        Initializes a BaseMetricAdapter with specific configurations for metric calculation.

        Args:
            name (str): Name of the metric, which will be used during logging as `<stage>/<name>_<log_level>`.
            metric_init_fn (callable): A function used to initialize metrics, should return a metric instance.
            input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
            stages (List[str]): The stages (e.g., 'train', 'val', 'test') at which the metric is active.
            metric_kwargs (Dict, optional): Additional keyword arguments to pass to metric_fn during `batch_update`.
            fill_nan: (bool): If True, when the required input_pos_args not appears in the input,
                instead of raise an exception, will fill nan instead.
        """
        super().__init__(name, input_pos_args, stages)

        if metric_kwargs is None:
            metric_kwargs = {}
        self.metric_kwargs = metric_kwargs

        self.metrics = torch.nn.ModuleDict({
            self.STAGE_PREFIX + p: metric_init_fn() for p in stages
        })

        self.batch_val = {self.STAGE_PREFIX + i: None for i in stages}

        self.fill_nan = fill_nan

    def batch_update(self, stage, **kwargs):
        """
        Update the metric value for a specific batch and stage.

        Args:
            stage (str): The current stage of the training/validation/testing process.
            **kwargs: Arbitrary keyword arguments containing the inputs needed for the metric computation.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')

        if stage not in self.stages:
            return
        if self.fill_nan:
            inputs = [kwargs.get(k, torch.nan) for k in self.input_pos_args]
        else:
            inputs = [kwargs.get(k) for k in self.input_pos_args]

        batch_val = self.metrics[self.STAGE_PREFIX + stage](*inputs, **self.metric_kwargs)

        self.batch_val[self.STAGE_PREFIX + stage] = batch_val

    def get_value(self, stage, level) -> Dict:
        """
        Retrieve the current value of the metric for a given stage and level.

        Args:
            stage (str): Current stage of the process.
            level (str): The level at which the metric value is required ('batch' or 'epoch').

        Returns:
            Dict: Metric value formatted as {f'{self.name}_{level}': value}.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')
        if level not in LEVELS:
            raise ValueError(f'level should be one of `{LEVELS}`')

        if stage not in self.stages:
            return {}

        if level == LEVEL_BATCH:
            value = self.batch_val[self.STAGE_PREFIX + stage]
        elif level == LEVEL_EPOCH:
            value = self.metrics[self.STAGE_PREFIX + stage].compute()
        else:
            raise ValueError(f'error level {level}')

        return {f'{self.name}_{level}': value}

    def reset(self, stage):
        """
        Reset the metric for a specified stage, clearing any stored values and reinitializing as necessary.

        Args:
            stage (str): Stage at which the metric should be reset.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')

        if stage not in self.stages:
            return

        self.metrics[self.STAGE_PREFIX + stage].reset()


class BaseMetaMetric(MGMetric):
    """
    Base class for meta-metrics that are updated at the end of each epoch and are typically used
    to aggregate or compare values from other BaseMetrics. These metrics are reset only once at the
    beginning of each run and are designed to provide higher-level insights into the performance
    across different stages.

    It is designed to be subclassed to implement complicated metric calculations.
    For typical use cases, consider using `BaseMetaMetricAdapter`.

    Pattern:
        - Updates: Each epoch, based on aggregated data from BaseMetrics.
        - Inputs: Takes values from other metrics as inputs (which is automatically handled by `MetricGroup`).
        - Logging: Logs metrics at the end of each epoch.
        - Resets: Resets once at the beginning of a run or experiment.

    Attributes:
        name (str): Name for the meta-metric, which will be used during logging as `<stage>/<name>`.
        input_pos_args (List[str]):  Ordered keys of input data from the inputs dictionary that the metric uses.
        stages (Iterable[str]): Stages (e.g., 'train', 'val', 'test') during which the meta-metric is active.
    """

    def __init__(self, name: str, input_pos_args: List[str], stages: Iterable[str]) -> None:
        """
        Args:
            name (str): Name of the metric, which will be used during logging as `<stage>/<name>`.
            input_pos_args (List[str]): Ordered keys of input data from the inputs dictionary that the metric uses.
            stages (Iterable[str]): The stages (e.g., 'train', 'val', 'test') at which the metric is active.
        """
        super().__init__()
        self.name = name
        self.input_pos_args = input_pos_args
        self.stages = stages

    @abc.abstractmethod
    def epoch_update(self, stage, **kwargs):
        """
        Update the meta-metric based on values provided by other metrics at the end of an epoch.

        Args:
            stage (str): Current stage of training/validation/testing.
            **kwargs: Metric values from other metrics as inputs.
        """
        pass

    @abc.abstractmethod
    def get_value(self, stage) -> Dict:
        """
        Retrieve the current value of the meta-metric.

        Args:
            stage (str): Current stage of the process.

        Returns:
            Dict: Meta-metric value formatted as {metric_name: value}.
        """
        pass

    @abc.abstractmethod
    def reset(self, stage):
        """
        Reset the meta-metric for a new run or experiment.

        Args:
            stage (str): Stage at which the reset should occur.
        """
        pass


class BaseMetaMetricAdapter(BaseMetaMetric):
    """
    Adapter for BaseMetaMetric that facilitates the initialization and dynamic configuration of metrics.
    This class extends BaseMetaMetric to support the creation of metric instances on-the-fly, based on the provided
    initialization function and additional metric-specific arguments.

    Pattern:
        - Updates: Each epoch, based on aggregated data from BaseMetrics.
        - Inputs: Takes values from other metrics as inputs (which is automatically handled by `MetricGroup`).
        - Logging: Logs metrics at the end of each epoch.
        - Resets: Resets once at the beginning of a run or experiment.


    Attributes:
        name (str): Name for the meta-metric, which will be used during logging as `<stage>/<name>`.
        input_pos_args (List[str]):  Ordered keys of input data from the inputs dictionary that the metric uses.
        stages (Iterable[str]): Stages (e.g., 'train', 'val', 'test') during which the meta-metric is active.
        metrics (torch.nn.ModuleDict): Contains instantiated meta-metrics for each stage.
        metric_kwargs (Dict): Additional keyword arguments to pass to metric_fn during `epoch_update`.
    """

    def __init__(self, name: str, metric_init_fn,
                 input_pos_args: List[str], stages: Iterable[str] = STAGES,
                 metric_kwargs: Dict = None, fill_nan: bool = True) -> None:
        """
        Args:
            name (str): Name for the meta-metric, which will be used during logging as `<stage>/<name>`.
            metric_init_fn (callable): A function used to initialize metrics, should return a metric instance.
            input_pos_args (List[str]):  Ordered keys of input data from the inputs dictionary that the metric uses.
            stages (Iterable[str]): Stages (e.g., 'train', 'val', 'test') during which the meta-metric is active.
            metric_kwargs (Dict, optional): Additional keyword arguments to pass to metric_fn during `epoch_update`.
            fill_nan: (bool): If True, when the required input_pos_args not appears in the input,
                instead of raise an exception, will fill nan instead.
        """
        super().__init__(name, input_pos_args, stages)

        if metric_kwargs is None:
            metric_kwargs = {}
        self.metric_kwargs = metric_kwargs

        self.metrics = torch.nn.ModuleDict({
            self.STAGE_PREFIX + p: metric_init_fn() for p in stages
        })

        self.fill_nan = fill_nan

    def epoch_update(self, stage, **kwargs):
        """
        Update the meta-metric based on values provided by other metrics at the end of an epoch.

        Args:
            stage (str): Current stage of training/validation/testing.
            **kwargs: Metric values from other metrics as inputs.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')

        if stage not in self.stages:
            return

        if self.fill_nan:
            inputs = [kwargs.get(k, torch.nan) for k in self.input_pos_args]
        else:
            inputs = [kwargs.get(k) for k in self.input_pos_args]

        batch_val = self.metrics[self.STAGE_PREFIX + stage](*inputs, **self.metric_kwargs)

    def get_value(self, stage) -> Dict:
        """
        Retrieve the current value of the meta-metric.

        Args:
            stage (str): Current stage of the process.

        Returns:
            Dict: Meta-metric value formatted as {metric_name: value}.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')

        if stage not in self.stages:
            return {}

        val = self.metrics[self.STAGE_PREFIX + stage].compute()

        return {f'{self.name}': val}

    def reset(self, stage):
        """
        Reset the meta-metric for a new run or experiment.

        Args:
            stage (str): Stage at which the reset should occur.
        """
        if stage not in STAGES:
            raise ValueError(f'stage should be one of `{STAGES}`')

        if stage not in self.stages:
            return

        self.metrics[self.STAGE_PREFIX + stage].reset()
