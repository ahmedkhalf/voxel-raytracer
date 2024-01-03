# Voxel Raytracer

A voxel is like a 3d pixel used in many fields and applications. It can also make for some interesting 3D art.

## TODO

- support changing of camera angle and position
- support different voxel materials
- support shadows and lights (emissive materials)
- support many models in the same scene
- avoid unnecessary voxel marching by using AABB collisions check with the voxel model
- try using compute shaders rather than fragment shaders, it could be beneficial though I need to research why
- render using multiple passes rather than a single pass to support lower-end GPUs, and generally allow for more samples per pixel
- voxel editor !? (a bit of a stretch)

## References

The following books, papers, and other references have been used in the making of this personal project:

- https://raytracing.github.io/books/RayTracingInOneWeekend.html
- http://www.cse.yorku.ca/~amana/research/grid.pdf
- https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#antialiasingandpseudorandomnumbergeneration/pseudorandomnumbergenerationinglsl

## Gallery

All of the following were rendered on a GTX 1050 TI Mobile with 256 samples per pixel and 50 max bounces.

![image](https://github.com/ahmedkhalf/voxel-raytracer/assets/36672196/07413f34-0d8f-40ad-90c2-1f58b6738ba8)

Rendered in 82.95 milliseconds.

![image](https://github.com/ahmedkhalf/voxel-raytracer/assets/36672196/4262e008-7be7-4476-9978-5ae49484b88b)

Rendered in 74.47 milliseconds.

![image](https://github.com/ahmedkhalf/voxel-raytracer/assets/36672196/a370eeae-4cbb-46fd-9b29-0294fd36d9d5)

Rendered in 516.46 milliseconds.
