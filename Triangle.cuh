#pragma once

#include "framework.h"
#include "Ray.cuh"
#include "Hit.h"
#include "Intersectable.cuh"

class Triangle : public Intersectable {
	vec3 p1, p2, p3;
	vec3 normal;
	Material* material;

	__host__ __device__ Triangle(vec3 _p1, vec3 _p2, vec3 _p3) {
		p1 = _p1; p2 = _p2; p3 = _p3;
		normal = normalize(cross((p2 - p1), (p3 - p1)));
	}

	__host__ __device__ size_t size() {
		return sizeof(Triangle);
	}

	__host__ __device__ Hit intersect(const Ray& ray) {
		Hit hit;

		float t = dot((p1 - ray.start), normal) / dot(ray.dir, normal);

		if (t < 0.0f) {
			return hit;
		}

		vec3 p = ray.start + ray.dir * t;
		if (!dot(cross((p2 - p1), (p - p1)), normal) > 0) {
			return hit;
		}
		if (!dot(cross((p3 - p2), (p - p2)), normal) > 0) {
			return hit;
		}
		if (!dot(cross((p1 - p3), (p - p3)), normal) > 0) {
			return hit;
		}
		hit.t = t;
		hit.position = p;
		hit.normal = normal;
		hit.material = material;
		return hit;
	}
};