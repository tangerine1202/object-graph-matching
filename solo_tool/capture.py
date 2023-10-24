import warnings
from pprint import pprint
import numpy as np

from .annotation import (
    Annotation,
    SemanticSegmentationAnnotation,
    InstanceSegmentationAnnotation,
    DepthAnnotation,
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
)


class Capture():
    def __init__(self, capture, root):
        self.root = root
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
                return SemanticSegmentationAnnotation(annotation, self.root)
            elif annotation_id == 'instance segmentation':
                return InstanceSegmentationAnnotation(annotation, self.root)
            elif annotation_id == 'Depth':
                return DepthAnnotation(annotation)
            elif annotation_id == 'bounding box':
                return BoundingBox2DAnnotation(annotation)
            elif annotation_id == 'bounding box 3D':
                return BoundingBox3DAnnotation(annotation)
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
