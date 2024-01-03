#version 430
#define SAMPLES_PER_PIXEL 256  // should prob be a uniform
#define MAX_DEPTH 50  // this too
#define MAX_MODELS 10


struct VoxelModel {
    vec3 position;
    ivec3 size;
};

uniform uvec2 iResolution;
uniform int numModels;
uniform VoxelModel models[MAX_MODELS];

layout(std430, binding = 0) buffer ssbo {  // model data
    vec4 data[];
};

out vec4 fragColor;


struct Ray {
    vec3 origin;
    vec3 direction;
};

struct HitRecord {
    vec3 p;
    float t;
    vec3 normal;
    vec3 albedo;  // should be replaced by material
};


// PCG random number generation, [0, 1] inclusive
// https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#antialiasingandpseudorandomnumbergeneration/pseudorandomnumbergenerationinglsl
uint rngState = iResolution.x * uint(gl_FragCoord.y) + uint(gl_FragCoord.x);
uint stepRandom(uint rngState) { return rngState * 747796405 + 1; }
float random(inout uint rngState) {
    rngState  = stepRandom(rngState);
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word      = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

bool voxelPresent(ivec3 voxOrigin, out vec3 albedo) {
    ivec3 modelSize = models[0].size;
    if (voxOrigin.x >= modelSize.x || voxOrigin.y >= modelSize.y || voxOrigin.z >= modelSize.z || voxOrigin.x < 0 || voxOrigin.y < 0 || voxOrigin.z < 0) {
        return false;
    }

    vec4 voxel = data[voxOrigin.x + voxOrigin.y * modelSize.x + voxOrigin.z * modelSize.x * modelSize.y];
    albedo = voxel.xyz;
    return voxel.a > 0;
}

bool voxelMarch(Ray ray, out HitRecord rec) {
    ivec3 voxOrigin = ivec3(floor(ray.origin));

    ivec3 posStepDir = ivec3(step(0.0, ray.direction));  // 0.0 <= rayDirection ? 1.0 : 0.0
    ivec3 stepDir = ivec3(posStepDir * 2 - 1);

    ivec3 justOut = ivec3(posStepDir * (models[0].size - 1) + stepDir);

    vec3 tMax = (posStepDir - (ray.origin - voxOrigin)) / ray.direction;
    vec3 tDelta = stepDir / ray.direction;

    ivec3 lastOrigin = voxOrigin;
    float tMin = 0;

    for (int i = 0; i < 1000; ++i) {
        bool present = voxelPresent(voxOrigin, rec.albedo);
        if (present) {
            rec.normal = lastOrigin - voxOrigin;
            rec.t = tMin;
            rec.p = ray.origin + tMin * ray.direction;

            return true;
        }

        lastOrigin = voxOrigin;
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                voxOrigin.x += stepDir.x;
                if (voxOrigin.x == justOut.x) { break; }
                tMin = tMax.x;
                tMax.x += tDelta.x;
            } else {
                voxOrigin.z += stepDir.z;
                if (voxOrigin.z == justOut.z) { break; }
                tMin = tMax.z;
                tMax.z += tDelta.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                voxOrigin.y += stepDir.y;
                if (voxOrigin.y == justOut.y) { break; }
                tMin = tMax.y;
                tMax.y += tDelta.y;
            } else {
                voxOrigin.z += stepDir.z;
                if (voxOrigin.z == justOut.z) { break; }
                tMin = tMax.z;
                tMax.z += tDelta.z;
            }
        }
    }

    return false;
}

bool groundHit(Ray ray, out HitRecord rec) {
    float t = (0 - ray.origin.y) / ray.direction.y;  // replace 0 with y position of ground, TODO make configurable
    if (t < 0 || t > 50) {  // 100 for now, TODO use interval object
        return false;
    }

    rec.normal = vec3(0.0, float(ray.direction.y < 0) * 2.0 - 1.0, 0.0);
    rec.t = t;
    rec.albedo = vec3(0.8, 0.8, 0.0);
    rec.p = ray.origin + ray.direction * t;

    return true;
}

// TODO improve this function lol
bool worldHit(Ray ray, out HitRecord rec) {
    HitRecord ground_rec;
    HitRecord vox_rec;

    bool ground = groundHit(ray, ground_rec);
    bool vox = voxelMarch(ray, vox_rec);

    if (ground && vox) {
        if (ground_rec.t < vox_rec.t) {
            rec = ground_rec;
            return true;
        } else {
            rec = vox_rec;
            return true;
        }
    }

    if (ground) {
        rec = ground_rec;
        return true;
    }

    if (vox) {
        rec = vox_rec;
        return true;
    }

    return false;
}

float randrange(float range_min, float range_max) {
    return range_min + (range_max - range_min) * random(rngState);
}

vec3 random_in_unit_sphere() {
    vec3 p = vec3(randrange(-1,1), randrange(-1,1), randrange(-1,1));
    for (int i = 0; i < 20; i++) {
        if (p.x * p.x + p.y * p.y + p.z * p.z < 1) {
            return p;
        }
    }
    return p;
}

vec3 random_unit_vector() {
    return normalize(random_in_unit_sphere());
}

Ray get_ray() {
    float image_width = iResolution.x;
    float image_height = iResolution.y;
    float aspect_ratio = image_width / image_height;

    vec3 center = vec3(models[0].size.x / 2, models[0].size.y / 2, models[0].size.z + 12);

    float focal_length = 1.0;
    float viewport_height = 2.0;
    float viewport_width = viewport_height * image_width / image_height;

    vec3 viewport_u = vec3(viewport_width, 0.0, 0.0);
    vec3 viewport_v = vec3(0.0, viewport_height, 0.0);

    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    vec3 viewport_lower_left = center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;

    vec3 pixel00_loc = viewport_lower_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    return Ray(
        center,
        normalize((pixel00_loc + (gl_FragCoord.x - 0.5 + random(rngState)) * pixel_delta_u
                               + (gl_FragCoord.y - 0.5 + random(rngState)) * pixel_delta_v) - center)
    );
}

vec3 get_color() {
    vec3 out_color = vec3(1, 1, 1);

    Ray ray = get_ray();

    HitRecord rec;
    for (int d = 0; d < MAX_DEPTH; d++) {
        if (worldHit(ray, rec)) {
            vec3 scatter_dir = rec.normal + random_unit_vector();

            // handle scatter_dir near zero case
            if (scatter_dir.x < 1e-8 && scatter_dir.y < 1e-8 && scatter_dir.z < 1e-8) {
                scatter_dir = rec.normal;
            }

            ray.origin = rec.p + scatter_dir * 0.001;
            ray.direction = scatter_dir;

            out_color *= rec.albedo;
        } else {
            float a = 0.5 * (ray.direction.y + 1.0);
            out_color *= mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), a);
            break;
        }
    }

    return out_color;
}

void main() {
    vec3 out_color = vec3(0, 0, 0);

    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
        out_color += get_color();
    }

    fragColor = vec4(sqrt(out_color / SAMPLES_PER_PIXEL), 1.0);
}
