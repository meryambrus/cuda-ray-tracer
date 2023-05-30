#pragma once

#include "Ray.cuh"
#include "Hit.cuh"
#include "Intersectable.cuh"


class Sphere : public Intersectable {
public:
	vec3 center;
	float radius;
	Material* material;

	__host__ __device__ Sphere(vec3 _center, float _radius, Material* _material = nullptr);

	__host__ __device__ Sphere(const Sphere& obj);

	__host__ __device__ size_t size();

	//Ray-Sphere intersection
	__host__ __device__ Hit intersect(const Ray& ray) override;

	__host__ static Sphere** createOnDevice(vec3 _center, float _radius, Material* _material = nullptr);

	__host__ static void deleteOnDevice(Sphere** dev_ptr);
};

__global__ void createSphereOnDevice(Sphere** ptr, Sphere sphere);

__global__ void deleteSphereOnDevice(Sphere** ptr);
