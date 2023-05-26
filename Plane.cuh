#pragma once

#include "Ray.cuh"
#include "Hit.h"

extern const float EPSILON;

class Plane;
__global__ void createPlaneOnDevice(Plane** ptr, Plane plane);
__global__ void deletePlaneOnDevice(Plane** ptr);

class Plane : public Intersectable{
public:
	vec3 point;
	vec3 normal;
	Material* material;
	float epsilon;

	__host__ Plane(vec3 _point, vec3 _normal, Material* _material = nullptr) {
		point = _point; normal = _normal; material = _material; epsilon = EPSILON;
	}

	__host__ __device__ size_t size() {
		return sizeof(Plane);
	}

	__host__ __device__ Hit intersect(const Ray& ray) {
		Hit hit;

		float NdotV = dot(normal, ray.dir);

		if (fabs(NdotV) < epsilon) {
			return hit;
		}

		float t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) {
			return hit;
		}

		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		hit.material = material;

		return hit;
	}

	__host__ static Plane** createOnDevice(vec3 _point, vec3 _normal, Material* _material = nullptr) {
		Plane** dev_ptr;
		cudaMalloc(&dev_ptr, sizeof(Plane*));

		createPlaneOnDevice<<<1, 1>>>(dev_ptr, Plane(_point, _normal, _material));
		return dev_ptr;
	}

	__host__ static void deleteOnDevice(Plane** dev_ptr) {
		deletePlaneOnDevice<<<1, 1>>>(dev_ptr);
	}
};

__global__ void createPlaneOnDevice(Plane** ptr, Plane plane) {
	(*ptr) = new Plane(plane);
}

__global__ void deletePlaneOnDevice(Plane** ptr) {
	delete* ptr;
}