#pragma once
#include "Sphere.cuh"

__host__ __device__ Sphere::Sphere(vec3 _center, float _radius, Material* _material) {
	center = _center;
	radius = _radius;
	material = _material;
}

__host__ __device__ Sphere::Sphere(const Sphere& obj) {
	this->center = obj.center;
	this->radius = obj.radius;
	this->material = obj.material;
}

__host__ __device__ size_t Sphere::size() {
	return sizeof(Sphere);
}

//Ray-Sphere intersection
__host__ __device__ Hit Sphere::intersect(const Ray& ray) {
	Hit hit;

	vec3 distance = ray.start - center;

	float a = dot(ray.dir, ray.dir);
	float b = dot(distance, ray.dir) * 2.0f;
	float c = dot(distance, distance) - radius * radius;

	float discr = b * b - 4.0f * a * c;

	if (discr < 0.0f) return hit;

	float sqrt_discr = sqrtf(discr);
	float t1 = (-b + sqrt_discr) / (2.0f * a);
	float t2 = (-b - sqrt_discr) / (2.0f * a);

	if (t1 <= 0) return hit;

	hit.t = (t2 > 0) ? t2 : t1;
	hit.position = ray.start + ray.dir * hit.t;
	hit.normal = (hit.position - center) * (1.0f / radius);
	hit.material = material;
	return hit;
}

__host__ Sphere** Sphere::createOnDevice(vec3 _center, float _radius, Material* _material) {
	Sphere** dev_ptr;
	cudaMalloc(&dev_ptr, sizeof(Sphere*));

	createSphereOnDevice << <1, 1 >> > (dev_ptr, Sphere(_center, _radius, _material));
	return dev_ptr;
}

__host__ void Sphere::deleteOnDevice(Sphere** dev_ptr) {
	deleteSphereOnDevice << <1, 1 >> > (dev_ptr);
}

__global__ void createSphereOnDevice(Sphere** ptr, Sphere sphere) {
	(*ptr) = new Sphere(sphere);
}

__global__ void deleteSphereOnDevice(Sphere** ptr) {
	delete* ptr;
}
