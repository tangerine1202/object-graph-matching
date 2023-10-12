import warnings
import os
import glob
import json
from pprint import pprint
import numpy as np
import pandas as pd
import cv2

from .annotation_definition import AnnotationDefinition, InstanceSegmentationAnnotationDefinition, BoundingBox2DAnnotationDefinition
from .annotation import Annotation, SemanticSegmentationAnnotation, InstanceSegmentationAnnotation, BoundingBox2DAnnotation
from .metric import Metric, GenericMetric


class Capture():
    def __init__(self, capture):
        self.capture = capture
        self.type = capture['@type']
        self.id = capture['id']
        self.description = capture['description']
        # camera pose
        self.position = np.array(capture['position'])
        self.rotation = np.array(capture['rotation'])
        self.velocity = np.array(capture['velocity'])
        self.acceleration = np.array(capture['acceleration'])
        # rgb capture
        self.filename = capture['filename']
        self.imageFormat = capture['imageFormat']
        self.dimension = np.array(capture['dimension'])
        self.projection = capture['projection']
        self.projectionMatrix = np.array(capture['matrix']).reshape(3, 3)
        annotations = [self._set_annotation_(annotation) for annotation in capture['annotations']]
        self.annotations = {annotation.id: annotation for annotation in annotations}

    def _set_annotation_(self, annotation):
        annotation_id = annotation['id']
        try:
            if annotation_id == 'semantic segmentation':
                return SemanticSegmentationAnnotation(annotation)
            elif annotation_id == 'instance segmentation':
                return InstanceSegmentationAnnotation(annotation)
            elif annotation_id == 'bounding box':
                return BoundingBox2DAnnotation(annotation)
            else:
                raise NotImplementedError
        except NotImplementedError:
            msg = f'Annotation id {annotation_id} not implemented'
            warnings.warn(msg)
            return Annotation(annotation)

    @property
    def camera_pose(self):
        return np.concatenate((self.position, self.rotation))

    @property
    def rgb_capture(self):
        return {
            'filename': self.filename,
            'imageFormat': self.imageFormat,
            'dimension': self.dimension,
            'projection': self.projection,
            'projectionMatrix': self.projectionMatrix,
        }


class Frame():
    def __init__(self, path, sequence_idx, frame_idx):
        self.sequence_name = self._get_sequence_name(sequence_idx)
        self.frame_name = self._get_frame_name(frame_idx)
        self.sequence_path = os.path.join(path, self.sequence_name)
        self.frame_path = os.path.join(path, self.sequence_name, self.frame_name)
        self.data = json.load(open(self.frame_path))
        self.frame = self.data['frame']
        self.sequence = self.data['sequence']
        self.step = self.data['step']
        self.timestamp = self.data['timestamp']
        self.captures = [Capture(capture) for capture in self.data['captures']]
        metrics = [self._set_metric_(metric) for metric in self.data['metrics']]
        self.metrics = {metric.id: metric for metric in metrics}

    def _set_metric_(self, metric):
        metric_id = metric['id']
        try:
            if metric_id == 'metadata':
                return GenericMetric(metric)
            else:
                raise NotImplementedError
        except NotImplementedError:
            msg = f'Metric id {metric_id} not implemented'
            # warnings.warn(msg)
            return Metric(metric)

    def _get_sequence_name(self, sequence_idx):
        return f'sequence.{sequence_idx}'

    def _get_frame_name(self, frame_idx):
        return f'step{frame_idx}.frame_data.json'


class Solo():
    def __init__(self, path):
        self.path = path
        self.metadata = json.load(open(os.path.join(path, 'metadata.json')))
        self.metric_definitions = json.load(open(os.path.join(path, 'metric_definitions.json')))
        self.sensor_definitions = json.load(open(os.path.join(path, 'sensor_definitions.json')))
        raw_annotation_definition = json.load(open(os.path.join(path, 'annotation_definitions.json')))
        annotation_definition = [self._set_annotation_def_(
            annotation_def) for annotation_def in raw_annotation_definition['annotationDefinitions']]
        self.annotation_definitions = {annotation_def.id: annotation_def for annotation_def in annotation_definition}

        self.num_sequences = self.metadata['totalSequences']
        self.num_frames = self.metadata['totalFrames']

    def _set_annotation_def_(self, annotation_def):
        annotation_def_id = annotation_def['id']
        try:
            if annotation_def_id == 'instance segmentation':
                return InstanceSegmentationAnnotationDefinition(annotation_def)
            elif annotation_def_id == 'bounding box':
                return BoundingBox2DAnnotationDefinition(annotation_def)
            else:
                raise NotImplementedError
        except NotImplementedError:
            msg = f'Annotation definition id {annotation_def_id} not implemented'
            warnings.warn(msg)
            return AnnotationDefinition(annotation_def)

    def frames(self):
        for sequence_idx in range(self.num_sequences):
            for frame_idx in range(self.num_frames):
                yield Frame(self.path, sequence_idx, frame_idx)


if __name__ == '__main__':
    SOLO_NAME = 'poisson3_vis'
    DATA_PATH = f'../output/SimpleOffice/{SOLO_NAME}'
    solo = Solo(DATA_PATH)

    for f in solo.frames():
        print(f.frame_name)
        cap = f.captures[0]
        metrics = f.metrics
        annos = cap.annotations

        seq_path = os.path.join(DATA_PATH, f.sequence_name)

        inst = annos['instance segmentation']
        inst.create_masks(seq_path)

        seg = annos['semantic segmentation']
        seg.create_masks(seq_path)

        bbox = annos['bounding box']

        meta = metrics['metadata']
        print(meta._env_metadata)
        # pprint(meta.flat_instances)
        print(meta._flat_instances_df)
