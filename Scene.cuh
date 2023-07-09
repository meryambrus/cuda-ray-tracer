#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Camera.cuh"
#include "Sphere.cuh"
#include "Plane.cuh"
#include "Triangle.cuh"
#include "Light.cuh"
#include "LaunchOptions.cuh"

#include "RenderFrameCuda.cuh"


extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

class Scene {
public:
	Camera* camera;
	
	std::vector<Light*> lights;

	std::vector<vec3> frame;
	vec3* dev_framebuffer;

	std::vector<Intersectable*> objects;
	std::vector<Intersectable**> dev_objects;

	Scene();

	void build();

	vec3* createFrameBufferOnDevice(int width, int height);

	void copyFrameBufferToDevice(int width, int height);

	template<typename T>
	T* copyObjectsToDeviceOfType(std::vector<T*>& objects) {

		if (objects.size() == 0) {
			return 0;
		}

		T* dev_objects;
		std::vector<T> host_objects;

		for (auto o : objects) {
			host_objects.push_back(*o);
		}

		cudaError_t cudaStatus = cudaMalloc((void**)&dev_objects, objects.size() * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}

		cudaStatus = cudaMemcpy(dev_objects, host_objects.data(), objects.size() * sizeof(T), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}

		return dev_objects;
	}

	void render(std::vector<vec4>& image);

	~Scene();
};
