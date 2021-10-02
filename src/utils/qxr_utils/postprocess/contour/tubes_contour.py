from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, validator
from skimage.morphology import skeletonize

from qxr_utils.image import transforms as tf
from qxr_utils.tag_config.constants import GeneralKeys

POINT = Tuple[int, int]


# borrowed from https://github.com/samuelcolvin/pydantic/issues/380
class _ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (Array,), {"__dtype__": t})


class Array(np.ndarray, metaclass=_ArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        dtype = getattr(cls, "__dtype__", None)
        if isinstance(dtype, tuple):
            dtype, shape = dtype
        else:
            shape = tuple()

        result = np.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        assert not shape or len(shape) == len(result.shape)  # ndmin guarantees this

        if any((shape[i] != -1 and shape[i] != result.shape[i]) for i in range(len(shape))):
            result = result.reshape(shape)
        return result


class TubeContour(BaseModel):
    # TODO - rewrite this class just using functions?
    mask: Array
    min_pix: int = 10
    seen: Optional[Array]
    branches: dict = {}

    @validator("seen", always=True)
    def check_mask(cls, v, values):
        seen = np.zeros_like(values["mask"])
        return seen

    @validator("min_pix")
    def check_min_pix(cls, v):
        assert v > 0, f"min_pix is {v}, which is not > 0"
        return v

    def is_valid(self, pt: POINT) -> bool:
        """
        checks if the point is valid and unseen during traversal
        Args:
            pt ([POINT]): a point

        Returns:
            bool : true if its a valid point
        """
        x, y = pt
        if x < 0 or x >= self.mask.shape[0]:
            return False
        if y < 0 or y >= self.mask.shape[1]:
            return False
        if self.seen[x, y]:
            return False
        return True

    def get_8_neighborhood(self, pt: POINT, img: Array) -> List[Optional[POINT]]:
        """
        returns 8 connected neighborhood of valid points

        Args:
            pt (POINT): a point
            img (Array): numpy array with the connected components, typically a
            skeletonized mask

        Returns:
            List[Optional[POINT]]: list of points in the neighborhood
        """
        out = []
        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                if not (i == 0 and j == 0):
                    new_pt = (pt[0] + i, pt[1] + j)
                    if self.is_valid(new_pt) and img[new_pt]:
                        out.append(new_pt)
        return out

    def traverse(self, point: POINT, root: POINT, img: Array):
        """
        recursively traverse from point with in a branch with a root.
        at the beginning, we use traverse(start, start)
        a
        Args:
            point (POINT): a point
            root (POINT): root of the branch in which the point is present
            img (Array): numpy array with the connected components, typically a
            skeletonized mask
        """
        self.seen[point] = 1
        self.branches[root].append(point)
        nbhd = self.get_8_neighborhood(point, img)
        #         print(root,point,nbhd)
        if len(nbhd) == 1:
            point = nbhd[0]
            self.traverse(point, root, img)
        if len(nbhd) > 1:
            #             print(point,nbhd)
            for pt in nbhd:
                root = pt
                if self.is_valid(root):
                    self.branches[root] = []
                    self.traverse(pt, root, img)

    def find_branches(self, start: POINT, img: Array):
        """
        finds all the branches by recursively traversing from the start
        Args:
            start (POINT): staring point
            img (Array): numpy array with the connected components, typically a
            skeletonized mask
        """
        self.branches[start] = []
        self.traverse(start, start, img)

    def draw_contours_et(
        self, shape: Optional[Tuple[int, int]] = None, offset: int = 0
    ) -> List[List[Optional[POINT]]]:
        """
        returns endotracheal tube contours
        Args:
            shape (Tuple[int, int], optional): Shape of the array on which
            points outputted will be drawn. Defaults to None.
            offset (int, optional): pixel offset by which points on the contour
            are translated horizontally. Defaults to 0.

        Returns:
            List[List[Optional[POINT]]]: list of list of points where each sub
            list is a contour
        """
        convex_hull = tf.get_convex_hull(self.mask)
        skeleton = tf.get_skeleton(convex_hull)

        if not shape:
            shape = self.mask.shape

        h, w = self.mask.shape
        h_new, w_new = shape
        out_pts = []
        branch_pts = []
        for i in range(0, len(skeleton) - 1):
            y1, x1 = skeleton[i]
            y2, x2 = skeleton[i + 1]
            x1 = int(x1 * w_new / w) + offset
            x2 = int(x2 * w_new / w) + offset
            y1 = int(y1 * h_new / h)
            y2 = int(y2 * h_new / h)

            if i == 0:
                branch_pts.append((x1, y1))
            branch_pts.append((x2, y2))

        out_pts.append(branch_pts)

        return out_pts

    def draw_contours_ngt(
        self, shape: Optional[Tuple[int, int]] = None, offset: int = 0
    ) -> List[List[Optional[POINT]]]:
        """
        returns nasogastric tube contours
        Args:
            shape (Tuple[int, int], optional): Shape of the array on which
            points outputted will be drawn. Defaults to None.
            offset (int, optional): pixel offset by which points on the contour
            are translated horizontally. Defaults to 0.

        Returns:
            List[List[Optional[POINT]]]: list of list of points where each sub
            list is a contour
        """
        if not shape:
            shape = self.mask.shape

        h, w = self.mask.shape
        h_new, w_new = shape
        out_pts = []

        img = skeletonize(self.mask.astype(bool))

        if np.sum(img) > 0:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                tf.cast_uint8(img.astype(int)), connectivity=8
            )
            label_largest = np.argsort(stats[:, -1])[-2]
            for label in range(1, nb_components):
                if label != label_largest:
                    if stats[label][-1] > self.min_pix:
                        component = output == label
                        ends = tf.end_points(component)
                        if len(ends) > 0:
                            start = min(ends, key=lambda x: x[0])
                            self.find_branches((int(start[0]), int(start[1])), img)

                if label == label_largest:
                    component = output == label
                    ends = tf.end_points(component)
                    if len(ends) > 0:
                        start = min(ends, key=lambda x: x[0])
                        self.find_branches((int(start[0]), int(start[1])), img)

            for key, pts in self.branches.items():
                branch_pts = []
                if len(pts) > self.min_pix:
                    for i in range(0, len(pts) - 1):
                        y1, x1 = pts[i]
                        y2, x2 = pts[i + 1]
                        x1 = int(x1 * w_new / w) + offset
                        x2 = int(x2 * w_new / w) + offset
                        y1 = int(y1 * h_new / h)
                        y2 = int(y2 * h_new / h)
                        if i == 0:
                            branch_pts.append((x1, y1))
                        branch_pts.append((x2, y2))
                out_pts.append(branch_pts)

        out_pts = [i for i in out_pts if len(i) > 0]
        return out_pts

    def draw_contours(
        self, tube_type: str, shape: Optional[Tuple[int, int]] = None, offset: int = 0
    ) -> List[List[Optional[POINT]]]:
        """
        returns contours depending on tube_type

        Args:
            tube_type (str): breathing tube or gastric tube
            shape (Optional[Tuple[int, int]]): [description]. Defaults to None.
            offset (int, optional): [description]. Defaults to 0.

        Returns:
            List[List[Optional[POINT]]]: [description]
        """
        # self._setup()
        dc = {
            GeneralKeys.breathingtube.value: self.draw_contours_et,
            GeneralKeys.gastrictube.value: self.draw_contours_ngt,
        }

        assert tube_type in dc, f"{tube_type} not supported"

        return dc[tube_type](shape, offset)
