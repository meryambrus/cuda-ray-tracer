#pragma once

#include "Hit.cuh"

struct Ray {
	vec3 start, dir;
	__host__ __device__ Ray(vec3 _start, vec3 _dir);
};
