# PackMetric

PackMetric is a lightweight, easy-to-use metric management tool built on top of `torchmetrics`. This tool simplifies the process of integrating, updating, and managing metrics, also enabling the separation of evaluation and computation code in your ML projects. PackMetric can be seamlessly intergrated into a lightning project, and it also supports configuration via Hydra, offering a flexible and scalable way to manage metrics throughout the lifecycle of a model.

## Features

- **DDP Support**: Ensures metrics work correctly under distributed training conditions through `torchmetrics`.
- **Seamless Integration with PyTorch Lightning**: Adds easily into the PyTorch Lightning workflow as a callback.
- **Code Separation**: Facilitates the separation of metric evaluation from computation, enhancing code modularity.
- **Hydra Configuration**: Supports using Hydra for dynamic configuration of metrics and their management.
- 
## Installation

Install PackMetric via pip with the following command:

```bash
pip install packmetric
```

## Usage

Using PackMetric in your projects involves constructing a `MetricGroup` and then integrating it within your PyTorch Lightning model, either with or without a callback. Here's how to do it:

### Constructing a MetricGroup

#### 1. Directly Through Code

You can create a `MetricGroup` directly in your code by instantiating it with your desired metrics from `torchmetrics`.

```python
import packmetric
from packmetric import MetricGroup, BaseMetricAdapter, BaseMetaMetricAdapter, STAGE_TRAIN, STAGE_VAL, STAGE_TEST
from packmetric.utils.template import MeanMetricAdapter
from torchmetrics import Accuracy, MaxMetric


def custom_metric(x, y, some_parameter):
    return x ** 2 + y * some_parameter - 1


# Define your metrics
accuracy = BaseMetricAdapter(name='acc',
                             metric_init_fn=lambda: Accuracy(task="multiclass", num_classes=4),
                             input_pos_args=['pred', 'target'],
                             stages=[STAGE_TRAIN, STAGE_VAL])
max_accuracy = BaseMetaMetricAdapter(name='max_acc',
                                     metric_init_fn=MaxMetric,
                                     input_pos_args=['acc'])
cm = MeanMetricAdapter('cm')

# Create a metric group
metric_group = MetricGroup(input_metrics=[accuracy, max_accuracy, cm])
```

#### 2. Through hydra config

An example config could be as bellow. For more details, please refer to hydra.

```yaml
metrics:
  accuracy:
    _target_: packmetric.BaseMetricAdapter
    name: acc
    metric_init_fn:
      _target_: torchmetrics.Accuracy
      task: "multiclass"
      num_classes: 4
      _partial_: true
    input_pos_args: ["pred", "target"]
    stages: ["train", "val"]

  max_accuracy:
    _target_: packmetric.BaseMetaMetricAdapter
    name: max_acc
    metric_init_fn:
      _target_: torchmetrics.MaxMetric
      _partial_: true
    input_pos_args: ["acc"]

  custom_metric:
    _target_: packmetric.utils.template.MeanMetricAdapter
    name: cm
```

```python
from typing import List

import hydra
from omegaconf import DictConfig

from packmetric import MGMetric, MetricGroup


def instantiate_metrics(metric_cfg: DictConfig) -> List[MGMetric]:
    """Instantiates metrics from config."""
    metrics: List[MGMetric] = []

    if not metric_cfg:
        print("Metric config is empty.")
        return metrics

    if not isinstance(metric_cfg, DictConfig):
        raise TypeError("Metric config must be a DictConfig!")

    for _, m_conf in metric_cfg.items():
        if isinstance(m_conf, DictConfig) and "_target_" in m_conf:
            print(f'Instantiating metric "{m_conf.name}"')
            metrics.append(hydra.utils.instantiate(m_conf))

    return metrics


metric_cfg = ...

metric_group = MetricGroup(instantiate_metrics(metric_cfg))

```

### Using MetricGroup


#### Dirctly Through Code

Once your `MetricGroup` is configured, you can integrate it directly into your training loops. Here’s an example of how to use `MetricGroup` to track metrics during training:

```python
n_epoch = ...
dataloaders = ...
model = ...

metric_group.reset(level=packmetric.LEVEL_RUN)

for epoch in range(n_epoch):
    for batch in dataloaders['train']:
        y_hat = model(batch.x)

        step_metrics = metric_group.batch_step(
            {'pred': y_hat, 'target': batch.y, 'cm': custom_metric(batch.x, batch.y, batch.some_parameter)},
            stage=STAGE_TRAIN
        )

    epoch_metrics = metric_group.epoch_step(STAGE_TRAIN)
    print(epoch_metrics)
```

#### As a lightning callback

```python
from packmetric.utils.lightning import LogMetricsCallback
from pytorch_lightning import Trainer

# Initialize your model and MetricGroup as shown above

# Add the PackMetric callback
metric_callback = LogMetricsCallback(metric_group=metric_group)

# Create a trainer and pass the callback
trainer = Trainer(callbacks=[metric_callback])
trainer.fit(model, train_dataloader)
```

