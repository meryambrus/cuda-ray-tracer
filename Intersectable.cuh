#pragma once

#include "Hit.cuh"
#include "Ray.cuh"

class Intersectable {
public:
	Intersectable** device_pointer = nullptr;

	__host__ __device__ Intersectable();

	__host__ __device__ virtual Hit intersect(const Ray& ray) = 0;
	__host__ __device__ virtual size_t size() = 0;
	__host__ __device__ ~Intersectable();
};
