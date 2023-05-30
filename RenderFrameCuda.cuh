#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Hit.cuh"
#include "Camera.cuh"
#include "Intersectable.cuh"
#include "ShapeData.cuh"

extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

__device__ Hit getFirstHit(Intersectable*** objects, int num_objects, const Ray& ray);

__device__ bool DirectionalLightShadowed(Intersectable*** objects, int num_objects, const Ray& ray);

__device__ bool floatEqual(const float& f1, const float& f2);

__device__ bool vec3Equal(const vec3& v1, const vec3& v2);

__device__ vec3 vec3Abs(const vec3& v);


__device__ vec3 trace(const Ray& ray, int depth);


__global__ void renderGPU(Camera camera, Intersectable*** objects, int num_objects, Light* lights, int num_lights, vec3* frameBuffer, int width, int height);
