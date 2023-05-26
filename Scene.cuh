#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math_functions.h"
#include "Hit.h"
#include "Sphere.cuh"
#include "Plane.cuh"
#include "Light.cuh"
#include "LaunchOptions.cuh"
#include "Triangle.cuh"
#include "RenderFrameCuda.h"


extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

class Scene {
public:
	Camera* camera;
	
	std::vector<Light*> lights;

	std::vector<vec3> frame;
	vec3* dev_framebuffer;

	std::vector<Intersectable**> dev_objects;

	Scene() {

	}

	void build() {
		camera = new Camera();
		camera->set(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), (60.0f/180.0f)*M_PI);
		camera->set(vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), (60.0f / 180.0f) * M_PI);

		printf("Camera up   : %.2f %.2f %.2f\n", camera->up.x, camera->up.y, camera->up.z);
		printf("Camera right: %.2f %.2f %.2f\n", camera->right.x, camera->right.y, camera->right.z);

		//spheres.push_back(new Sphere(vec3(1.0f, 0.1f, 0.2f), 0.2f));
		//spheres.push_back(new Sphere(vec3(1.0f, 0.0f, -0.0f), 0.2f));

		dev_objects.push_back((Intersectable**)Sphere::createOnDevice(vec3(1.0f, 0.1f, 0.4f), 0.2f));
		dev_objects.push_back((Intersectable**)Sphere::createOnDevice(vec3(1.0f, 0.0f, -0.1f), 0.2f));

		//planes.push_back(new Plane(vec3(0.0f, -0.2f, 0.0f), vec3(0.0f, 1.0f, 0.0f)));

		dev_objects.push_back((Intersectable**)Plane::createOnDevice(vec3(0.0f, -0.2f, 0.0f), vec3(0.0f, 1.0f, 0.0f)));

		dev_framebuffer = createFrameBufferOnDevice(WINDOW_WIDTH, WINDOW_HEIGHT);
		frame.reserve(WINDOW_WIDTH * WINDOW_HEIGHT);

		lights.push_back(new Light(vec3(1.2f, 0.0f, -0.6f), vec3(0.0f, 0.0f, 1.0f), LightType::POINT));
		lights.push_back(new Light(vec3(0.0f, -1.0f, 1.0f), vec3(0.0f, 0.0f, 0.6f), LightType::DIRECTIONAL));
		lights.push_back(new Light(vec3(0.0f, -1.0f, -1.0f), vec3(0.6f, 0.0f, 0.0f), LightType::DIRECTIONAL));
		lights.push_back(new Light(vec3(-1.0f, -1.0f, 0.0f), vec3(0.0f, 0.6f, 0.0f), LightType::DIRECTIONAL));
	}

	vec3* createFrameBufferOnDevice(int width, int height) {
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

	void copyFrameBufferToDevice(int width, int height) {
		cudaError_t cudaStatus = cudaMemcpy(dev_framebuffer, frame.data(), frame.size() * sizeof(vec3), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}
	}

	float* createTBufferOnDevice(int width, int height) {
		float* dev_t_buffer;

		cudaError_t cudaStatus = cudaMalloc((void**)&dev_t_buffer, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}

		std::vector<float> ts;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				ts.push_back(-1.0f);
			}
		}

		cudaStatus = cudaMemcpy(dev_t_buffer, ts.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}

		return dev_t_buffer;
	}

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

	LaunchOptions getKernelLaunchOptions(int width, int height) {
		int blockSize = getGreatestCommonFactor(width, height);
		while (blockSize * blockSize > 1024) {
			blockSize /= 2;
		}

		dim3 threadsPerBlock(blockSize, blockSize);
		//printf("threadsPerBlock: %d, %d\n", threadsPerBlock.x, threadsPerBlock.y);

		dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
		//printf("numBlocks: %d, %d\n", numBlocks.x, numBlocks.y);

		

		return LaunchOptions(threadsPerBlock, numBlocks);
	}

	void renderNew(std::vector<vec4>& image) {

		Light* dev_lights = copyObjectsToDeviceOfType<Light>(lights);

		LaunchOptions lo = getKernelLaunchOptions(WINDOW_WIDTH, WINDOW_HEIGHT);

		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		

		Intersectable*** dev_ptrs;

		cudaMalloc((void**)&dev_ptrs, dev_objects.size() * sizeof(Intersectable**));
		cudaMemcpy(dev_ptrs, dev_objects.data(), dev_objects.size() * sizeof(Intersectable**), cudaMemcpyHostToDevice);

		//renderWithPolymorphism<<<lo.numBlocks, lo.threadsPerBlock>>>(*camera, dev_ptrs, dev_objects.size(), dev_lights, lights.size(), dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);
		dim3 numBlocks(WINDOW_WIDTH / 32 + 1, WINDOW_HEIGHT / 32 + 1);
		dim3 threadsPerBlock(32, 32);
		renderWithPolymorphism<<<numBlocks, threadsPerBlock>>>(*camera, dev_ptrs, dev_objects.size(), dev_lights, lights.size(), dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);


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

	~Scene() {
		cudaFree(dev_framebuffer);
		
	}
};


/*
void render(std::vector<vec4>& image) {
	vec3* dev_framebuffer;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_framebuffer, sizeof(vec3) * WINDOW_WIDTH * WINDOW_HEIGHT);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	std::vector<vec3> frame;
	for (int i = 0; i < WINDOW_WIDTH; i++) {
		for (int j = 0; j < WINDOW_HEIGHT; j++) {
			frame.push_back(vec3(0.0f, 0.0f, 0.0f));
		}
	}

	cudaStatus = cudaMemcpy(dev_framebuffer, frame.data(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(vec3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	int blockSize = getGreatestCommonFactor(WINDOW_WIDTH, WINDOW_HEIGHT);
	while (blockSize * blockSize > 1024) {
		blockSize /= 2;
	}

	dim3 threadsPerBlock(blockSize, blockSize);
	printf("threadsPerBlock: %d, %d\n", threadsPerBlock.x, threadsPerBlock.y);

	dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);
	printf("numBlocks: %d, %d\n", numBlocks.x, numBlocks.y);

	//renderFrame<<<numBlocks, threadsPerBlock>>>(*camera, sphere, dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "renderFrame failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSynchronize failed!");
	}

	cudaStatus = cudaMemcpy(frame.data(), dev_framebuffer, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(vec3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Host failed!");
	}

	printf("Frame size %d\n", frame.size());

	for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
		image[i] = vec4(frame[i], 1.0f);
	}
}
*/

/*
void renderMultipleObjects(std::vector<vec4>& image) {
		
		vec3* dev_framebuffer;

		cudaError_t cudaStatus = cudaMalloc((void**)&dev_framebuffer, sizeof(vec3) * WINDOW_WIDTH * WINDOW_HEIGHT);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}

		std::vector<vec3> frame;
		for (int i = 0; i < WINDOW_WIDTH; i++) {
			for (int j = 0; j < WINDOW_HEIGHT; j++) {
				frame.push_back(vec3(0.0f, 0.0f, 0.0f));
			}
		}

		cudaStatus = cudaMemcpy(dev_framebuffer, frame.data(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(vec3), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}
		
	



		int blockSize = getGreatestCommonFactor(WINDOW_WIDTH, WINDOW_HEIGHT);
		while (blockSize * blockSize > 1024) {
			blockSize /= 2;
		}

		dim3 threadsPerBlock(blockSize, blockSize);
		printf("threadsPerBlock: %d, %d\n", threadsPerBlock.x, threadsPerBlock.y);

		dim3 numBlocks(WINDOW_WIDTH / threadsPerBlock.x, WINDOW_HEIGHT / threadsPerBlock.y);
		printf("numBlocks: %d, %d\n", numBlocks.x, numBlocks.y);

		//renderFrameMultiple<<<numBlocks, threadsPerBlock>>>(*camera, device_pointers[0], device_pointers[1], device_pointers.size(), dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "renderFrame failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSynchronize failed!\n");
		}

		cudaStatus = cudaMemcpy(frame.data(), dev_framebuffer, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(vec3), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy to Host failed!\n");
		}

		printf("Frame size %d\n", frame.size());

		for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; i++) {
			image[i] = vec4(frame[i], 1.0f);
		}
	}
*/

/*

		std::vector<Intersectable*> object_dev_ptr;
		for (int i = 0; i < objects.size(); i++) {
			Intersectable* dev_ptr;

			cudaStatus = cudaMalloc((void**)&dev_ptr, objects[i]->size());
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed! i = %d\n", i);
			}
			//&(*objects[i])
			cudaStatus = cudaMemcpy(dev_ptr, objects[i], objects[i]->size(), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed! i = %d\n", i);
			}
			printf("Sphere[%d] = %d\n", i, dev_ptr);
			object_dev_ptr.push_back(dev_ptr);
		}


		Intersectable** dev_vector_ptr;

		cudaStatus = cudaMalloc((void**)&dev_vector_ptr, sizeof(Intersectable*) * object_dev_ptr.size());
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!\n");
		}

		cudaStatus = cudaMemcpy(dev_vector_ptr, object_dev_ptr.data(), object_dev_ptr.size() * sizeof(Intersectable*), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
		}
		printf("dev_vector_ptr == %d\n", dev_vector_ptr);
*/

/*


		//renderShapesTemplated<Sphere><<<lo.numBlocks, lo.threadsPerBlock>>>(*camera, dev_spheres, spheres.size(), dev_lights, lights.size(), dev_t_buffer, dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);

		//renderShapesTemplated<Plane><<<lo.numBlocks, lo.threadsPerBlock>>>(*camera, dev_planes, planes.size(), dev_lights, lights.size(), dev_t_buffer, dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);
*/

//spheres.push_back(new Sphere(vec3(1.0f, 0.0f, 0.0f), 0.3f, nullptr));

/*
Sphere** dev_sphere;
cudaMalloc(&dev_sphere, sizeof(Sphere*));

createSphereOnDevice<<<1, 1>>>(dev_sphere, Sphere(vec3(0.5f, 0.5f, 0.5f), 0.25f));

printDetails<<<1, 1>>>((Intersectable**)dev_sphere);

		float* dev_t_buffer = createTBufferOnDevice(WINDOW_WIDTH, WINDOW_HEIGHT);

		Sphere* dev_spheres = copyObjectsToDeviceOfType<Sphere>(spheres);

		Plane* dev_planes = copyObjectsToDeviceOfType<Plane>(planes);

		//renderWithLights<<<lo.numBlocks, lo.threadsPerBlock>>>(*camera, data, dev_framebuffer, WINDOW_WIDTH, WINDOW_HEIGHT);

				//copyFrameBufferToDevice(WINDOW_WIDTH, WINDOW_HEIGHT);

		//Sphere* dev_spheres = copyObjectsToDeviceOfType<Sphere>(spheres);

		//Plane* dev_planes = copyObjectsToDeviceOfType<Plane>(planes);

				//cudaFree(dev_spheres);
		//cudaFree(dev_planes);

		std::vector<Sphere*> spheres;
	std::vector<Plane*> planes;
	//Sphere sphere = Sphere(vec3(1.0f, 0.0f, 0.0f), 0.3f, nullptr);

	for (auto o : spheres) { delete o; }
		for (auto o : planes) { delete o; }
*/

// TODO szétbontani alap alakzatokra, pixelData device pointert adni a kerneleknek, ami tárolja a pixel színét és a legkisebb t-t
//camera->rotate(0.02f, vec3(0.0f, 1.0f, 0.0f));
		//camera->rotate(0.02f, camera->up);