from typing import Any, Callable, Optional

import torch
import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from packmetric import LEVEL_EPOCH, LEVEL_RUN, MetricGroup, STAGE_TEST, STAGE_TRAIN, STAGE_VAL


class LogMetricsCallback(Callback):
    """
    A callback for PyTorch Lightning that handles the automatic logging and resetting of metrics
    for training, validation, and testing stages. This callback uses a MetricGroup instance to
    manage and update metrics based on the outputs of each batch and epoch.

    Attributes:
        metric_group (MetricGroup): The group of metrics that will be updated and logged.
        batch_size_fn (Callable[[Any], int], optional): A function that takes a batch as input
            and returns the batch size. If None, batch size handling will be ignored.
    """

    def __init__(self, metric_group: MetricGroup, batch_size_fn: Optional[Callable] = None):
        """
        Initializes the LogMetricsCallback with a MetricGroup and an optional batch size function.

        Args:
            metric_group (MetricGroup): The metric group that will be used for logging and updating metrics.
            batch_size_fn (Callable[[Any], int], optional): A function to determine the batch size from
                the input batch for more accurate logging. Defaults to None, in which case batch size logging is ignored.
        """
        self.metric_group = metric_group
        self.batch_size_fn = (lambda batch: None) if batch_size_fn is None else batch_size_fn

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Resets the metrics at the start of training.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The training module.
        """
        self.metric_group.reset(level=LEVEL_RUN)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any, batch_idx: int) -> None:
        """
        Updates and logs metrics at the end of each training batch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The lightning module.
            outputs (STEP_OUTPUT): Outputs from the training step.
            batch (Any): The input batch.
            batch_idx (int): Index of the current batch.
        """
        batch_size = self.batch_size_fn(batch)

        if outputs is None:
            outputs = {}
        elif isinstance(outputs, torch.Tensor):
            outputs = {'output': outputs}

        step_metrics = self.metric_group.batch_step(outputs, stage=STAGE_TRAIN)

        pl_module.log_dict(step_metrics, prog_bar=True, batch_size=batch_size)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Logs metrics and resets them at the end of each training epoch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The lightning module.
        """
        epoch_metrics = self.metric_group.epoch_step(stage=STAGE_TRAIN)

        pl_module.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, stages=STAGE_TRAIN)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Updates and logs metrics at the end of each validation batch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The training module.
            outputs (STEP_OUTPUT): Outputs from the validation step.
            batch (Any): The input batch.
            batch_idx (int): Index of the current batch.
        """
        batch_size = self.batch_size_fn(batch)

        if outputs is None:
            outputs = {}
        elif isinstance(outputs, torch.Tensor):
            outputs = {'output': outputs}

        # update and log metrics
        step_metrics = self.metric_group.batch_step(outputs, stage=STAGE_VAL)

        pl_module.log_dict(step_metrics, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Logs metrics and resets them at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The lightning module.
        """
        epoch_metrics = self.metric_group.epoch_step(stage=STAGE_VAL)
        pl_module.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, stages=STAGE_VAL)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Updates and logs metrics at the end of each test batch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The lightning module.
            outputs (STEP_OUTPUT): Outputs from the test step.
            batch (Any): The input batch.
            batch_idx (int): Index of the current batch.
        """
        batch_size = self.batch_size_fn(batch)

        if outputs is None:
            outputs = {}
        elif isinstance(outputs, torch.Tensor):
            outputs = {'output': outputs}

        step_metrics = self.metric_group.batch_step(outputs, stage=STAGE_TEST)
        pl_module.log_dict(step_metrics, prog_bar=True, batch_size=batch_size)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Logs metrics and resets them at the end of each test epoch.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The lightning module.
        """
        epoch_metrics = self.metric_group.epoch_step(stage=STAGE_TEST)
        pl_module.log_dict(epoch_metrics)

        self.metric_group.reset(level=LEVEL_EPOCH, stages=STAGE_TEST)
