#pragma once
#include "Ray.cuh"

__host__ __device__ Ray::Ray(vec3 _start, vec3 _dir) {
	start = _start; dir = _dir;
}
