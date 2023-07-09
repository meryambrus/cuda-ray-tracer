#pragma once
#include "Scene.cuh"

Scene::Scene() {

}

void Scene::build() {
	camera = new Camera();
	camera->set(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), (60.0f / 180.0f) * M_PI);
	//camera->set(vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), (60.0f / 180.0f) * M_PI);

	printf("Camera up   : %.2f %.2f %.2f\n", camera->up.x, camera->up.y, camera->up.z);
	printf("Camera right: %.2f %.2f %.2f\n", camera->right.x, camera->right.y, camera->right.z);

	Material* mat1 = new Material(vec3(0.2f, 0.0f, 0.0f), vec3(0.6f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), 100.0f);
	Material* mat2 = new Material(vec3(0.0f, 0.2f, 0.0f), vec3(0.0f, 0.6f, 0.0f), vec3(1.0f, 1.0f, 1.0f), 32.0f);
	Material* mat3 = new Material(vec3(0.2f, 0.2f, 0.2f), vec3(0.6f, 0.6f, 0.6f));

	objects.push_back(new Sphere(vec3(1.0f, 0.1f, 0.4f), 0.2f, mat1));
	objects.push_back(new Sphere(vec3(1.0f, 0.0f, -0.1f), 0.2f, mat2));
	objects.push_back(new Plane(vec3(0.0f, -0.2f, 0.0f), vec3(0.0f, 1.0f, 0.0f), mat3));

	for (auto o : objects) {
		dev_objects.push_back(o->device_pointer);
	}

	dev_framebuffer = createFrameBufferOnDevice(WINDOW_WIDTH, WINDOW_HEIGHT);
	frame.reserve(WINDOW_WIDTH * WINDOW_HEIGHT);

	//lights.push_back(new Light(vec3(1.2f, 0.0f, -0.6f), vec3(0.0f, 1.0f, 0.0f), LightType::POINT));
	lights.push_back(new Light(vec3(0.5f, 0.2f, 0.0f), vec3(1.0f, 1.0f, 1.0f), LightType::POINT));
	lights.push_back(new Light(vec3(0.0f, -1.0f, 1.0f), vec3(0.5f, 0.5f, 0.5f), LightType::DIRECTIONAL));
	//lights.push_back(new Light(vec3(0.0f, -1.0f, -1.0f), vec3(0.6f, 0.0f, 0.0f), LightType::DIRECTIONAL));
	//lights.push_back(new Light(vec3(-1.0f, -1.0f, 0.0f), vec3(0.0f, 0.6f, 0.0f), LightType::DIRECTIONAL));
}

vec3* Scene::createFrameBufferOnDevice(int width, int height) {
	vec3* dev_framebuffer;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_framebuffer, sizeof(vec3) * width * height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}

	std::vector<vec3> frame;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			frame.push_back(vec3(0.0f, 0.0f, 0.0f));
		}
	}

	cudaStatus = cudaMemcpy(dev_framebuffer, frame.data(), width * height * sizeof(vec3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	return dev_framebuffer;
}

void Scene::copyFrameBufferToDevice(int width, int height) {
	cudaError_t cudaStatus = cudaMemcpy(dev_framebuffer, frame.data(), frame.size() * sizeof(vec3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
}

void Scene::render(std::vector<vec4>& image) {

	Light* dev_lights = copyObjectsToDeviceOfType<Light>(lights);

	long timeStart = glutGet(GLUT_ELAPSED_TIME);


	Intersectable*** dev_ptrs;

	cudaMalloc((void**)&dev_ptrs, dev_objects.size() * sizeof(Intersectable**));
	cudaMemcpy(dev_ptrs, dev_objects.data(), dev_objects.size() * sizeof(Intersectable**), cudaMemcpyHostToDevice);


	dim3 numBlocks(WINDOW_WIDTH / 32 + 1, WINDOW_HEIGHT / 32 + 1);
	dim3 threadsPerBlock(32, 32);
	renderGPU<<<numBlocks, threadsPerBlock>>>(*camera, dev_ptrs, dev_objects.size(), dev_lights, lights.size(), dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);


	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderFrame failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSynchronize failed!");
	}

	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Kernel time: %d milliseconds\n", (timeEnd - timeStart));

	cudaStatus = cudaMemcpy(frame.data(), dev_framebuffer, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(vec3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Host failed!");
	}

	for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
		image[i] = vec4(frame[i], 1.0f);
	}
}

Scene::~Scene() {
	cudaFree(dev_framebuffer);
	for (auto o : objects) {
		delete o;
	}
}