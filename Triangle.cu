#pragma once
#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle(vec3 _p1, vec3 _p2, vec3 _p3, Material* _material) {
	p1 = _p1; p2 = _p2; p3 = _p3;
	normal = normalize(cross((p2 - p1), (p3 - p1)));
}

__host__ __device__ size_t Triangle::size() {
	return sizeof(Triangle);
}

__host__ __device__ Hit Triangle::intersect(const Ray& ray) {
	Hit hit;

	float t = dot((p1 - ray.start), normal) / dot(ray.dir, normal);

	if (t < 0.0f) {
		return hit;
	}

	vec3 p = ray.start + ray.dir * t;
	if (dot(cross((p2 - p1), (p - p1)), normal) < 0) {
		return hit;
	}
	if (dot(cross((p3 - p2), (p - p2)), normal) < 0) {
		return hit;
	}
	if (dot(cross((p1 - p3), (p - p3)), normal) < 0) {
		return hit;
	}
	hit.t = t;
	hit.position = p;
	hit.normal = normal;
	hit.material = material;
	return hit;
}

__host__ Triangle** Triangle::createOnDevice(vec3 _p1, vec3 _p2, vec3 _p3, Material* _material) {
	Triangle** dev_ptr;
	cudaMalloc(&dev_ptr, sizeof(Triangle*));

	createTriangleOnDevice << <1, 1 >> > (dev_ptr, Triangle(_p1, _p2, _p3, _material));
	return dev_ptr;
}

__host__ void Triangle::deleteOnDevice(Triangle** dev_ptr) {
	deleteTriangleOnDevice << <1, 1 >> > (dev_ptr);
}

__global__ void createTriangleOnDevice(Triangle** ptr, Triangle triangle) {
	(*ptr) = new Triangle(triangle);
}

__global__ void deleteTriangleOnDevice(Triangle** ptr) {
	delete* ptr;
}
