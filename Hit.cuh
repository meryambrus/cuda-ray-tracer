#pragma once

#include "Material.cuh"

struct Hit {
	float t;
	vec3 position, normal;

	Material* material = nullptr;

	__host__ __device__ Hit();
};
