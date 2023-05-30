#pragma once
#include "Plane.cuh"

__host__ Plane::Plane(vec3 _point, vec3 _normal, Material* _material) {
	point = _point; normal = _normal; material = _material; epsilon = EPSILON;
}

__host__ __device__ size_t Plane::size() {
	return sizeof(Plane);
}

__host__ __device__ Hit Plane::intersect(const Ray& ray) {
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

__host__ Plane** Plane::createOnDevice(vec3 _point, vec3 _normal, Material* _material) {
	Plane** dev_ptr;
	cudaMalloc(&dev_ptr, sizeof(Plane*));

	createPlaneOnDevice << <1, 1 >> > (dev_ptr, Plane(_point, _normal, _material));
	return dev_ptr;
}

__host__ void Plane::deleteOnDevice(Plane** dev_ptr) {
	deletePlaneOnDevice << <1, 1 >> > (dev_ptr);
}

__global__ void createPlaneOnDevice(Plane** ptr, Plane plane) {
	(*ptr) = new Plane(plane);
}

__global__ void deletePlaneOnDevice(Plane** ptr) {
	delete* ptr;
}