import warnings
import os
import numpy as np
import pandas as pd

class Metric():
  def __init__(self, metric):
    self.metric = metric
    self.type = self.metric['@type']
    self.id = self.metric['id']
    self.sensorId = self.metric['sensorId']
    self.annotationId = self.metric['annotationId']
    self.description = self.metric['description']


class GenericMetric(Metric):
  def __init__(self, metric):
    super().__init__(metric)
    # TODO: will there be more than one value?
    self._values = metric['values'][0]
    self._instances = self._values['instances']

    # environment-related metadata
    env_metadata = {}
    for labeler, metadata in self._values.items():
      if labeler == 'instances': continue
      env_metadata.update({labeler: metadata})
    self._env_metadata = env_metadata

    # instance-related metadata
    self._flat_instances = []
    for instance in self._instances:
      instance_metadata = {
        'instanceId': int(instance['instanceId']),
      }
      for labeler, metadata in instance.items():
        if labeler == 'instanceId': continue
        for key, value in metadata.items():
          prefix_key = f'{labeler}_{key}'
          instance_metadata.update({prefix_key: value})
      self._flat_instances.append(instance_metadata) 
    self._flat_instances_df = pd.DataFrame(self._flat_instances)
  
  @property
  def env_metadata(self):
    return self._env_metadata.copy()
  @property
  def instances(self):
    return self._instances.copy()
  @property
  def instances_df(self):
    return self._flat_instances_df.copy()