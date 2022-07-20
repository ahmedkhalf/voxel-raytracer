import math
import time
from math import copysign, floor
from typing import Iterator, Tuple

import pygame
from pyvox.parser import VoxParser
from tqdm import trange


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


class Renderer:
    def __init__(self, screen_size) -> None:
        self.screen_weight, self.screen_height = screen_size

        m1 = VoxParser("structure.vox").parse()
        self.img = m1.to_dense_rgba()
        self.size_x, self.size_y, self.size_z, _ = self.img.shape

        self.raycaster = VoxelRaycaster(self.size_x, self.size_y, self.size_z)

    def screen_to_world(self, x, y):
        return (
            (x / (self.screen_weight - 1)) * self.size_x,
            (y / (self.screen_height - 1)) * self.size_y,
        )

    def get(self, x: int, y: int) -> Tuple[int, int, int]:
        new_x, new_y = self.screen_to_world(x, y)
        new_z = -5

        origin = (new_x, new_y, new_z)
        direction = (0, 0, 1)

        for x, y, z, _ in self.raycaster.cast(origin, direction):
            if x not in range(self.size_x):
                continue
            if y not in range(self.size_y):
                continue
            if z not in range(self.size_z):
                continue

            r, g, b, a = self.img[x, y, z]
            if a == 255:
                return r, g, b

        return 0, 0, 0


class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 640, 400
        self.renderer = Renderer(self.size)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            self.size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._running = True

        # render the image
        self._display_surf.fill((0, 0, 0))
        start_time = time.time()
        for x in trange(self.width):
            for y in range(self.height):
                color = self.renderer.get(x, y)
                self._display_surf.set_at((x, y), color)
        print("--- %s seconds to render ---" % (time.time() - start_time))

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() is False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
