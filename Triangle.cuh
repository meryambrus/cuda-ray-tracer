#pragma once

#include "framework.h"
#include "Ray.cuh"
#include "Hit.cuh"
#include "Intersectable.cuh"

class Triangle : public Intersectable {
public:
	vec3 p1, p2, p3;
	vec3 normal;
	Material* material;

	__host__ __device__ Triangle(vec3 _p1, vec3 _p2, vec3 _p3, Material* _material = nullptr);

	__host__ __device__ size_t size();

	__host__ __device__ Hit intersect(const Ray& ray);

	__host__ static Triangle** createOnDevice(vec3 _p1, vec3 _p2, vec3 _p3, Material* _material = nullptr);

	__host__ static void deleteOnDevice(Triangle** dev_ptr);
};

__global__ void createTriangleOnDevice(Triangle** ptr, Triangle triangle);

__global__ void deleteTriangleOnDevice(Triangle** ptr);