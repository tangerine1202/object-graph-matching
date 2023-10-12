import pandas as pd

class AnnotationDefinition():
  def __init__(self, annotation_def):
    self.annotation_def = annotation_def
    self.type = self.annotation_def['@type']
    self.id = self.annotation_def['id']
    self.description = self.annotation_def['description']
    if 'spec' in self.annotation_def.keys():
      self.spec = self.annotation_def['spec']
    else:
      self.spec = None

    self.id2name = {}
    self.name2id = {}

class InstanceSegmentationAnnotationDefinition(AnnotationDefinition):
  def __init__(self, annotation_def):
    super().__init__(annotation_def)
    if self.spec is None:
      self.id2name = {}
      self.name2id = {}
    else:
      self.id2name = {label['label_id']: label['label_name'] for label in self.spec}
      self.name2id = {label['label_name']: label['label_id'] for label in self.spec}

class BoundingBox2DAnnotationDefinition(AnnotationDefinition):
  def __init__(self, annotation_def):
    super().__init__(annotation_def)
    if self.spec is None:
      self.id2name = {}
      self.name2id = {}
    else:
      self.id2name = {label['label_id']: label['label_name'] for label in self.spec}
      self.name2id = {label['label_name']: label['label_id'] for label in self.spec}