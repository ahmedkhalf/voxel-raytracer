from typing import Iterator, Tuple
from math import floor, copysign
import math


class VoxelRaycaster:
    def __init__(self, width: int, height: int, depth: int) -> None:
        self._width = width
        self._height = height
        self._depth = depth

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth

    @property
    def dim(self):
        return (self._width, self._height, self._depth)

    def set_dim(self, width: int, height: int, depth: int) -> None:
        self._width = width
        self._height = height
        self._depth = depth

    def cast(self, origin, direction) -> Iterator[Tuple[int, int, int, float]]:
        """
        Preconditions:
            - 0 <= x < self.width
            - 0 <= y < self.height
            - 0 <= z < self.depth
        """
        x = floor(origin[0])
        y = floor(origin[1])
        z = floor(origin[2])

        dir_x, dir_y, dir_z = direction

        step_x = int(copysign(1, dir_x))
        step_y = int(copysign(1, dir_y))
        step_z = int(copysign(1, dir_z))

        positive_step_x = step_x > 0
        positive_step_y = step_y > 0
        positive_step_z = step_z > 0

        just_out_x = positive_step_x * (self.width - 1) + step_x
        just_out_y = positive_step_y * (self.height - 1) + step_y
        just_out_z = positive_step_z * (self.depth - 1) + step_z

        # Unlike c, python does not implicitly set division by 0 to inf
        if dir_x != 0:
            t_max_x = (positive_step_x - (origin[0] - x)) / dir_x
            t_delta_x = step_x / dir_x
        else:
            t_max_x = math.inf
            t_delta_x = math.inf

        if dir_y != 0:
            t_max_y = (positive_step_y - (origin[1] - y)) / dir_y
            t_delta_y = step_y / dir_y
        else:
            t_max_y = math.inf
            t_delta_y = math.inf

        if dir_z != 0:
            t_max_z = (positive_step_z - (origin[1] - z)) / dir_z
            t_delta_z = step_y / dir_z
        else:
            t_max_z = math.inf
            t_delta_z = math.inf

        while True:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    yield x, y, z, t_max_x
                    x += step_x
                    if x == just_out_x:
                        return
                    t_max_x += t_delta_x
                else:
                    yield x, y, z, t_max_z
                    z += step_z
                    if z == just_out_z:
                        return
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    yield x, y, z, t_max_y
                    y += step_y
                    if y == just_out_y:
                        return
                    t_max_y += t_delta_y
                else:
                    yield x, y, z, t_max_z
                    z += step_z
                    if z == just_out_z:
                        return
                    t_max_z += t_delta_z
