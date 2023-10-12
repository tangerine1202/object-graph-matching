import warnings
import os
import glob
import json
import numpy as np
import pandas as pd
import cv2

class Annotation():
  def __init__(self, annotation):
    self.annotation = annotation
    self.type = self.annotation['@type']
    self.id = self.annotation['id']
    self.sensorId = self.annotation['sensorId']
    self.description = self.annotation['description']

  @property
  def instances(self):
    pass
  @property
  def instances_df(self):
    return pd.DataFrame()
  @property
  def values(self):
    pass
  @property
  def values_df(self):
    return pd.DataFrame()
  def create_masks(self, root):
    return


class SemanticSegmentationAnnotation(Annotation):
  def __init__(self, annotation):
    super().__init__(annotation)
    self.filename = self.annotation['filename']
    self.imageFormat = self.annotation['imageFormat']
    self.dimension = np.array(self.annotation['dimension'])
    self.has_instance = 'instances' in self.annotation.keys()
    self._instances = self.annotation['instances'] if self.has_instance else []
    self._instances_df = pd.DataFrame(self._instances)

  @property
  def instances(self):
    return self._instances.copy()

  @property
  def instances_df(self):
    return self._instances_df.copy()

  def create_masks(self, root):
    self.mask_dir = '.'.join(self.filename.split('.')[:-1])
    mask_dir_path = os.path.join(root, self.mask_dir)
    if not os.path.exists(mask_dir_path):
      os.mkdir(mask_dir_path)

    if not self.has_instance:
      msg = f'No instances to create masks for {self.filename}'
      warnings.warn(msg)
      return 

    img_path = os.path.join(root, self.filename)
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    for instance in self._instances:
      label = instance['labelName']
      color = np.array(instance['pixelValue'])
      color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
      lower = color_hsv
      upper = color_hsv
      mask = cv2.inRange(img_hsv, lower, upper)
      mask_filename = f'{label}.png'
      mask_path = os.path.join(mask_dir_path, mask_filename)
      cv2.imwrite(mask_path, mask)
      # update instance with filename
      instance.update({
        'mask_filename': mask_filename,
        # 'mask_path': mask_path,
      })
    # reset instances_df to include filenames
    self._instances_df = pd.DataFrame(self._instances)
  


class InstanceSegmentationAnnotation(Annotation):
  def __init__(self, annotation):
    super().__init__(annotation)
    self.filename = self.annotation['filename']
    self.imageFormat = self.annotation['imageFormat']
    self.dimension = np.array(self.annotation['dimension'])
    self.has_instance = 'instances' in self.annotation.keys()
    self._instances = self.annotation['instances'] if self.has_instance else []
    self._instances_df = pd.DataFrame(self._instances)
  
  @property
  def instances(self):
    return self._instances.copy()

  @property
  def instances_df(self):
    return self._instances_df.copy()

  def create_masks(self, root):
    self.mask_dir = '.'.join(self.filename.split('.')[:-1])
    mask_dir_path = os.path.join(root, self.mask_dir)
    if not os.path.exists(mask_dir_path):
      os.mkdir(mask_dir_path)

    if not self.has_instance:
      msg = f'No instances to create masks for {self.filename}'
      warnings.warn(msg)
      return 

    img_path = os.path.join(root, self.filename)
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    for instance in self._instances:
      instance_id = instance['instanceId']
      label_name = instance['labelName']
      color = np.array(instance['color'])
      color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
      lower = color_hsv
      upper = color_hsv
      mask = cv2.inRange(img_hsv, lower, upper)
      mask_filename = f'{instance_id}_{label_name}.png'
      mask_path = os.path.join(mask_dir_path, mask_filename)
      cv2.imwrite(mask_path, mask)
      # update instance with filename
      instance.update({
        'mask_filename': mask_filename,
        # 'mask_path': mask_path,
      })
    # reset instances_df to include filenames
    self._instances_df = pd.DataFrame(self._instances)


class BoundingBox2DAnnotation(Annotation):
  def __init__(self, annotation):
    super().__init__(annotation)
    self.has_instance = 'values' in self.annotation.keys()
    self._values = self.annotation['values'] if self.has_instance else []
    self._values_df = self._prepare_values_df_(self._values) if self.has_instance else pd.DataFrame()
  
  @property
  def values(self):
    return self._values.copy()
  
  @property
  def values_df(self):
    return self._values_df.copy()

  def _prepare_values_df_(self, values):
    values_df = pd.DataFrame(values)
    x0y0 = values_df['origin'].apply(pd.Series).rename(columns={0: 'x0', 1: 'y0'})
    wh = values_df['dimension'].apply(pd.Series).rename(columns={0: 'w', 1: 'h'})
    cxcy = pd.DataFrame(x0y0.values + wh.values / 2, columns=['cx', 'cy'], index=x0y0.index)
    values_df = values_df.drop(['origin', 'dimension'], axis=1)
    values_df = pd.concat([values_df, x0y0, wh, cxcy], axis=1)
    return values_df
