#pragma once
#include "Plane.cuh"

__host__ __device__ Plane::Plane(vec3 _point, vec3 _normal, Material* _material) {
	point = _point; normal = normalize(_normal); epsilon = EPSILON; epsilon = 1e-6;

	material = _material;

	/// Constructor will be called on device too,
	/// which means it would get stuck in a loop
	/// creating new objects until the stack overflows.
	/// When the host version of this is compiled, the macro
	/// __CUDA_ARCH__ won't be defined, while on the other
	/// hand, when compiling the device version, it will be.
#if !defined(__CUDA_ARCH__)
	device_pointer = (Intersectable**)createOnDevice(point, normal, *_material);
#endif
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

__host__ __device__ Plane::~Plane()
{
	deleteOnDevice((Plane**)device_pointer);
}

__host__ Plane** Plane::createOnDevice(vec3 _point, vec3 _normal, Material _material) {
	Plane** dev_ptr;
	cudaMalloc(&dev_ptr, sizeof(Plane*));

	createPlaneOnDevice << <1, 1 >> > (dev_ptr, _point, _normal, _material);
	return dev_ptr;
}

__host__ void Plane::deleteOnDevice(Plane** dev_ptr) {
	deletePlaneOnDevice << <1, 1 >> > (dev_ptr);
}

__global__ void createPlaneOnDevice(Plane** ptr, Plane plane) {
	(*ptr) = new Plane(plane);
}

__global__ void createPlaneOnDevice(Plane** ptr, vec3 _point, vec3 _normal, Material _material) {
	Material* mat = new Material(_material);

	(*ptr) = new Plane(_point, _normal, mat);
}

__global__ void deletePlaneOnDevice(Plane** ptr) {
	delete (*ptr)->material;
	delete* ptr;
}