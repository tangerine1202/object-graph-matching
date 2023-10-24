import argparse
import warnings
import os
import glob
import shutil
import json
from pprint import pprint
import numpy as np
import pandas as pd
import cv2

from .capture import Capture
from .annotation_definition import (
    AnnotationDefinition,
    InstanceSegmentationAnnotationDefinition,
    BoundingBox2DAnnotationDefinition,
    BoundingBox3DAnnotationDefinition
)
from .metric import Metric, GenericMetric


class Solo():
    def __init__(self, path, output_dir='data', is_reorganized=True, move=False):
        self.path = path
        self.output_dir = output_dir
        self.output_path = os.path.join(self.path, self.output_dir)
        self.metadata = json.load(open(os.path.join(path, 'metadata.json')))
        self.metric_definitions = json.load(open(os.path.join(path, 'metric_definitions.json')))
        self.sensor_definitions = json.load(open(os.path.join(path, 'sensor_definitions.json')))
        raw_annotation_definition = json.load(open(os.path.join(path, 'annotation_definitions.json')))
        annotation_definition = [self._set_annotation_def_(
            annotation_def) for annotation_def in raw_annotation_definition['annotationDefinitions']]
        self.annotation_definitions = {annotation_def.id: annotation_def for annotation_def in annotation_definition}

        self.num_sequences = self.metadata['totalSequences']
        self.num_frames = self.metadata['totalFrames']
        self.abbr2suffix = {
            'inst_seg': 'camera.instance segmentation.png',
            'sem_seg': 'camera.semantic segmentation.png',
            'depth': 'camera.Depth.exr',
            'rgb': 'camera.png',
            'meta': 'frame_data.json',
        }
        if not is_reorganized:
            self.reorganize(move=move)

    def reorganize(self, move=False):
        if os.path.exists(self.output_path):
            warnings.warn(f'{self.output_path} already exists')
        os.makedirs(self.output_path, exist_ok=True)

        for abbr in self.abbr2suffix.keys():
            os.makedirs(os.path.join(self.output_path, abbr), exist_ok=True)

        try:
            for seq_idx in range(self.num_sequences):
                seq_name = self._get_sequence_name(seq_idx)
                seq_path = os.path.join(self.path, seq_name)
                # get all frame paths
                file_paths = {
                    abbr: glob.glob(os.path.join(seq_path, f'*.{suffix}')) for abbr, suffix in self.abbr2suffix.items()
                }
                # save to output dir
                for abbr, paths in file_paths.items():
                    for path in paths:
                        step = os.path.basename(path).split('.')[0]
                        ext = os.path.basename(path).split('.')[-1]
                        filename = f'{step}.{ext}'
                        new_path = os.path.join(self.output_path, abbr, filename)
                        if move:
                            shutil.move(path, new_path)
                        else:
                            shutil.copy2(path, new_path)
        except Exception as e:
            warnings.warn(f'Error in reorganizing {seq_name}.')
            if move:
                warnings.warn(f'Files moved to {self.output_path} may be incomplete. Please check.')
            raise e

    def _get_sequence_name(self, sequence_idx):
        return f'sequence.{sequence_idx}'

    def _set_annotation_def_(self, annotation_def):
        annotation_def_id = annotation_def['id']
        try:
            if annotation_def_id == 'instance segmentation':
                return InstanceSegmentationAnnotationDefinition(annotation_def)
            elif annotation_def_id == 'bounding box':
                return BoundingBox2DAnnotationDefinition(annotation_def)
            elif annotation_def_id == 'bounding box 3D':
                return BoundingBox3DAnnotationDefinition(annotation_def)
            else:
                raise NotImplementedError
        except NotImplementedError:
            msg = f'Annotation definition id {annotation_def_id} not implemented'
            # warnings.warn(msg)
            return AnnotationDefinition(annotation_def)

    def frames(self):
        for step_idx in range(self.num_frames):
            filename = f'step{step_idx}'
            yield Frame(self.output_path, filename, self.abbr2suffix)

    def __len__(self):
        return self.num_frames


class Frame():
    def __init__(self, root, filename, abbr2suffix):
        self.abbr2suffix = abbr2suffix
        self.meta = json.load(open(os.path.join(root, 'meta', f'{filename}.json')))

        self.step = self.meta['step']
        self.sequence = self.meta['sequence']

        self.captures = [Capture(capture, root) for capture in self.meta['captures']]
        metrics = [self._set_metric_(metric) for metric in self.meta['metrics']]
        self.metrics = {metric.id: metric for metric in metrics}
        self.img_paths = {
            abbr: os.path.join(root, abbr, f'step{self.step}.{suffix.split(".")[-1]}') for abbr, suffix in self.abbr2suffix.items()
        }

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to solo output')
    parser.add_argument('--output_dir', type=str, default='data', help='name of the reorganized output directory')
    parser.add_argument('--reorganized', action='store_true', help='whether the data has been reorganized')
    parser.add_argument('--move', action='store_true', help='move files instead of copying')
    args = parser.parse_args()

    solo = Solo(args.path, args.output_dir, is_reorganized=args.reorganized, move=args.move)

    for f in solo.frames():
        cap = f.captures[0]
        metrics = f.metrics
        annos = cap.annotations
        semseg = annos['semantic segmentation']
        instseg = annos['instance segmentation']

    #     bbox = annos['bounding box']

    #     meta = metrics['metadata']
    #     print(meta._env_metadata)
    #     # pprint(meta.flat_instances)
    #     print(meta._flat_instances_df)
