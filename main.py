from __future__ import annotations

from math import sqrt, inf, floor
from dataclasses import dataclass
from abc import ABC
from random import random, uniform
from sys import stderr


class Vector3:
    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0):
        self._val = [a, b, c]

    @property
    def x(self):
        return self._val[0]

    @property
    def y(self):
        return self._val[1]

    @property
    def z(self):
        return self._val[2]

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __add__(self, other: Vector3 | float | int):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __sub__(self, other: Vector3 | float | int):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def __mul__(self, other: Vector3 | float | int):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return NotImplemented

    def __truediv__(self, other: Vector3 | float | int):
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, float) or isinstance(other, int):
            return Vector3(self.x / other, self.y / other, self.z / other)
        else:
            return NotImplemented

    def __radd__(self, other: float):
        return self.__add__(other)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __iadd__(self, other: Vector3 | float | int):
        if isinstance(other, Vector3):
            self._val[0] += other.x
            self._val[1] += other.y
            self._val[2] += other.z
        elif isinstance(other, float) or isinstance(other, int):
            self._val[0] += other
            self._val[1] += other
            self._val[2] += other
        else:
            return NotImplemented
        return self

    def length(self):
        return sqrt(self.length_squared())

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def near_zero(self):
        e = 1e-8
        return self.x < e and self.y < e and self.z < e

    @staticmethod
    def dot(u: Vector3, v: Vector3):
        return u.x * v.x + u.y * v.y + u.z * v.z

    @staticmethod
    def cross(u: Vector3, v: Vector3):
        return Vector3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        )

    @staticmethod
    def unit_vector(vec: Vector3):
        return vec / vec.length()

    @staticmethod
    def reflect(v: Vector3, normal: Vector3):
        return v - 2 * Vector3.dot(v, normal) * normal

    @staticmethod
    def random():
        return Vector3(random(), random(), random())

    @staticmethod
    def randrange(a: float, b: float):
        return Vector3(uniform(a, b), uniform(a, b), uniform(a, b))

    @staticmethod
    def random_in_unit_sphere():
        while True:
            p = Vector3.randrange(-1, 1)
            if p.length_squared() < 1:
                return p

    @staticmethod
    def random_unit_vector():
        return Vector3.unit_vector(Vector3.random_in_unit_sphere())

    @staticmethod
    def random_on_hemisphere(normal: Vector3):
        on_unit_sphere = Vector3.random_unit_vector()
        if Vector3.dot(on_unit_sphere, normal) > 0.0:
            return on_unit_sphere
        return -on_unit_sphere

    def __repr__(self):
        return f"[ {self.x}  {self.y}  {self.z} ]"


class Interval:
    def __init__(self, minimum: float | None = None, maximum: float | None = None):
        self.min = minimum if minimum is not None else inf
        self.max = maximum if maximum is not None else -inf

    def __contains__(self, item: float | int):
        if isinstance(item, float) or isinstance(item, int):
            return self.min <= item <= self.max
        else:
            return NotImplemented

    def surrounds(self, item: float | int):
        return self.min < item < self.max

    def clamp(self, x: float):
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x


@dataclass
class Ray:
    origin: Vector3
    direction: Vector3

    def at(self, t: float):
        return self.origin + t * self.direction


@dataclass
class ScatterOut:
    scatter: bool
    attenuation: Vector3
    scattered: Ray


class Material(ABC):
    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterOut:
        raise NotImplementedError


class Lambertian(Material):
    def __init__(self, albedo: Vector3):
        self.albedo = albedo

    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterOut:
        scatter_direction = rec.normal + Vector3.random_unit_vector()

        if scatter_direction.near_zero():
            scatter_direction = rec.normal

        scattered = Ray(rec.p, scatter_direction)
        return ScatterOut(True, self.albedo, scattered)


class Metal(Material):
    def __init__(self, albedo: Vector3, fuzz: float):
        self.albedo = albedo
        self.fuzz = fuzz

    def scatter(self, r_in: Ray, rec: HitRecord) -> ScatterOut:
        reflected = Vector3.reflect(Vector3.unit_vector(r_in.direction), rec.normal)
        scattered = Ray(rec.p, reflected + self.fuzz * Vector3.random_unit_vector())
        return ScatterOut(Vector3.dot(scattered.direction, rec.normal) > 0, self.albedo, scattered)


@dataclass
class HitRecord:
    p: Vector3
    mat: Material
    t: float
    normal: Vector3 = None
    front_face: bool = None

    def set_face_normal(self, r: Ray, outward_normal: Vector3):
        """NOTE: outward_normal assumed to be normalized"""
        self.front_face = Vector3.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


@dataclass
class HitOut:
    hit: bool
    rec: HitRecord = None


class Hittable(ABC):
    # out HitRecord
    def hit(self, r: Ray, ray_t: Interval) -> HitOut:
        raise NotImplementedError


class HittableList(Hittable):
    def __init__(self):
        self._objects: list[Hittable] = []

    def add(self, obj: Hittable):
        self._objects.append(obj)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitOut:
        out_hit = HitOut(False)
        closest_so_far = ray_t.max

        for obj in self._objects:
            if (obj_out_hit := obj.hit(r, Interval(ray_t.min, closest_so_far))).hit:
                closest_so_far = obj_out_hit.rec.t
                out_hit = obj_out_hit

        return out_hit


class Sphere(Hittable):
    def __init__(self, center: Vector3, radius: float, material: Material):
        self._center = center
        self._radius = radius
        self._material = material

    def hit(self, r: Ray, ray_t: Interval) -> HitOut:
        oc = r.origin - self._center
        a = r.direction.length_squared()
        half_b = Vector3.dot(oc, r.direction)
        c = oc.length_squared() - self._radius * self._radius

        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return HitOut(False)
        sqrtd = sqrt(discriminant)

        root = (-half_b - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (-half_b + sqrtd) / a
            if not ray_t.surrounds(root):
                return HitOut(False)

        p = r.at(root)
        outward_normal = (p - self._center) / self._radius
        hit_rec = HitRecord(t=root, p=p, mat=self._material)
        hit_rec.set_face_normal(r, outward_normal)

        return HitOut(True, hit_rec)


class Ground(Hittable):
    def __init__(self, y: float, material: Material):
        self._y = y
        self._material = material

    def hit(self, r: Ray, ray_t: Interval) -> HitOut:
        if r.direction.y == 0:
            return HitOut(False)

        t = (self._y - r.origin.y) / r.direction.y
        if not ray_t.surrounds(t):
            return HitOut(False)

        normal = Vector3(0.0, float(r.direction.y < 0) * 2.0 - 1.0, 0.0)
        hit_rec = HitRecord(t=t, p=r.at(t), mat=self._material)
        hit_rec.set_face_normal(r, normal)

        return HitOut(True, hit_rec)


class Camera:
    def __init__(self, aspect_ratio: float = 1.0, image_width: int = 100,
                 samples_per_pixel: int = 10, max_depth: int = 10):
        self._aspect_ratio = aspect_ratio
        self._image_width = image_width
        self._samples_per_pixel = samples_per_pixel
        self._max_depth = max_depth

        self._image_height = floor(image_width / aspect_ratio)
        self._image_height = 1 if self._image_height < 1 else self._image_height

        self._center = Vector3(0, 0, 0)

        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * self._image_width / self._image_height

        viewport_u = Vector3(viewport_width, 0.0, 0.0)
        viewport_v = Vector3(0.0, -viewport_height, 0.0)

        self._pixel_delta_u = viewport_u / self._image_width
        self._pixel_delta_v = viewport_v / self._image_height

        viewport_upper_left = self._center - \
            Vector3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2

        self._pixel00_loc = viewport_upper_left + 0.5 * (self._pixel_delta_u + self._pixel_delta_v)

    def get_ray(self, i: int, j: int) -> Ray:
        pixel_center = self._pixel00_loc + (i * self._pixel_delta_u) + (j * self._pixel_delta_v)
        pixel_sample = pixel_center + self.pixel_sample_square()

        return Ray(self._center, pixel_sample - self._center)

    def pixel_sample_square(self) -> Vector3:
        px = -0.5 + random()
        py = -0.5 + random()
        return px * self._pixel_delta_u + py * self._pixel_delta_v

    @staticmethod
    def ray_color(r: Ray, depth: int, world: Hittable) -> Vector3:
        if depth <= 0:
            return Vector3()

        if (hit_out := world.hit(r, Interval(0.001, inf))).hit:
            if (scatter_out := hit_out.rec.mat.scatter(r, hit_out.rec)).scatter:
                return scatter_out.attenuation * \
                       Camera.ray_color(scatter_out.scattered, depth - 1, world)
            return Vector3()

        unit_direction = Vector3.unit_vector(r.direction)
        a = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - a) * Vector3(1.0, 1.0, 1.0) + a * Vector3(0.5, 0.7, 1.0)

    def render(self, world: Hittable):
        print("P3")
        print(self._image_width, self._image_height)
        print(255)

        for j in range(self._image_height):
            print("\rScanlines remaining:", (self._image_height - j), "",
                  end="", flush=True, file=stderr)

            for i in range(self._image_width):
                pixel_color = Vector3()
                for _ in range(self._samples_per_pixel):
                    pixel_color += self.ray_color(self.get_ray(i, j), self._max_depth, world)

                # write pixel to file
                pixel_color = pixel_color / float(self._samples_per_pixel)
                pixel_color = Vector3(
                    sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z)
                )

                intensity = Interval(0.000, 0.999)
                r = floor(256 * intensity.clamp(pixel_color.x))
                g = floor(256 * intensity.clamp(pixel_color.y))
                b = floor(256 * intensity.clamp(pixel_color.z))

                # print(f"{int(pixel_color.x)} {int(pixel_color.y)} {int(pixel_color.z)}")
                print(f"{r} {g} {b}")
        print("\rDone.                 ", file=stderr)


def main():
    # mat_ground = Metal(Vector3(0.8, 0.8, 0.0), 0)
    mat_ground = Metal(Vector3(1.0, 1.0, 1.0), 0)
    mat_sphere = Lambertian(Vector3(0.7, 0.3, 0.3))

    ground = Ground(-0.5, mat_ground)
    sphere = Sphere(Vector3(0.0, 0.0, -1.0), 0.5, mat_sphere)

    world = HittableList()
    world.add(ground)
    world.add(sphere)

    cam = Camera(
        aspect_ratio=16 / 9,
        image_width=400,
        samples_per_pixel=100,
        max_depth=50
    )
    cam.render(world)


if __name__ == "__main__":
    main()
