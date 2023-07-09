#pragma once

#include "Ray.cuh"
#include "Hit.cuh"
#include "Material.cuh"
#include "Intersectable.cuh"

//extern const float EPSILON;

class Plane : public Intersectable{
public:
	vec3 point;
	vec3 normal;
	//device pointer to material
	Material* material;
	float epsilon;

	__host__ __device__ Plane(vec3 _point, vec3 _normal, Material* _material);

	__host__ __device__ size_t size();

	__host__ __device__ Hit intersect(const Ray& ray);

	__host__ __device__ ~Plane();

	__host__ static Plane** createOnDevice(vec3 _point, vec3 _normal, Material _material);

	__host__ static void deleteOnDevice(Plane** dev_ptr);
};

__global__ void createPlaneOnDevice(Plane** ptr, Plane plane);

__global__ void createPlaneOnDevice(Plane** ptr, vec3 _point, vec3 _normal, Material _material);

__global__ void deletePlaneOnDevice(Plane** ptr);