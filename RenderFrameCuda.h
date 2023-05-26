#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Hit.h"
#include "Camera.cuh"
#include "Intersectable.cuh"
#include "ShapeData.cuh"

extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

__device__ Hit getFirstHit(Intersectable*** objects, int num_objects, const Ray& ray) {
	Hit firstHit;

	for (int i = 0; i < num_objects; i++) {
		Hit hit = (*objects[i])->intersect(ray);

		if (hit.t >= 0.0f && (firstHit.t < 0.0f || hit.t < firstHit.t)) {
			firstHit = hit;
		}
	}

	if (firstHit.t >= 0.0f) {
		if (dot(ray.dir, firstHit.normal) < 0) {
			firstHit.normal = firstHit.normal * (-1);
		}
	}
	return firstHit;
}

__device__ bool DirectionalLightShadowed(Intersectable*** objects, int num_objects, const Ray& ray) {
	Hit hit;

	for (int i = 0; i < num_objects; i++) {
		hit = (*objects[i])->intersect(ray);
		if (hit.t >= 0.0f) {
			return true;
		}
	}
	return false;
}

__device__ bool floatEqual(const float& f1, const float& f2) {
	return (abs(f1 - f2) < 1e-4);
}

__device__ bool vec3Equal(const vec3& v1, const vec3& v2) {
	if (!floatEqual(v1.x, v2.x)) return false;
	if (!floatEqual(v1.y, v2.y)) return false;
	if (!floatEqual(v1.z, v2.z)) return false;
	return true;
}

__device__ vec3 vec3Abs(const vec3& v) {
	return vec3(abs(v.x), abs(v.y), abs(v.z));
}



__device__ vec3 trace(const Ray& ray, int depth) {
	vec3 weight = vec3(1.0f, 1.0f, 1.0f);

	vec3 outColor = vec3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < depth; i++) {

	}
}





__global__ void renderWithPolymorphism(Camera camera, Intersectable*** objects, int num_objects, Light* lights, int num_lights, vec3* frameBuffer, int width, int height) {

	if ((blockIdx.x * blockDim.x) + threadIdx.x > width || (blockIdx.y * blockDim.y) + threadIdx.y > height) return;

	float epsilon = 1e-4;
	int depth = 4;

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	int index = j * width + i;

	Hit firstHit = getFirstHit(objects, num_objects, ray);

	if (firstHit.t >= 0.0f) {

		vec3 color = vec3(0.6f, 0.6f, 0.6f) * dot(ray.dir, firstHit.normal);
		
		for (int k = 0; k < num_lights; k++) {
			Light& light = lights[k];
			Hit lightHit;
			if (light.isDirectional) {
				Ray lightRay = Ray(firstHit.position + (-light.pos_dir) * epsilon, -light.pos_dir);

				//Hit lightHit = getDirectionalLightHit(objects, num_objects, lightRay);

				if (!DirectionalLightShadowed(objects, num_objects, lightRay)) {
					color = color + light.color * dot(light.pos_dir, firstHit.normal);
				}
			}
			else {
				Ray lightRay = Ray(light.pos_dir, normalize(firstHit.position - light.pos_dir));

				Hit lightHit = getFirstHit(objects, num_objects, lightRay);

				if (lightHit.t > 0.0f && vec3Equal(firstHit.position, lightHit.position)) {

					float distance_falloff = length(light.pos_dir - lightHit.position);

					float intensity = 1.0f / (1.0f + distance_falloff) * (1.0f + distance_falloff);
					
					color = color + light.color  * dot(lightHit.normal, lightRay.dir) * intensity;
				}
			}
			
		}
		
		frameBuffer[index] = color;

	}
	else {
		//vec3(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f)
		frameBuffer[index] = vec3(0.521568f, 0.807843f, 0.921568f);
	}
}





/*
__global__ void printDetails(Intersectable** ptr) {
	//Ray ray = Ray(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
	//printf("center x: %.2f y: %.2f z: %.2f, radius: %.2f\n", ptr->center.x, ptr->center.y, ptr->center.z, ptr->radius);
	//ptr->intersect(ray);
	printf("hghgffgs\n");
	(*ptr)->test();
}*/


/*
__global__ void renderFrameMultiple(Camera camera, Intersectable* object1, Intersectable* object2, int numObjects, vec3* framebuffer, int width, int height) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	Hit firstHit;

	printf("object1 = %d\n", object1->size());
	printf("object2 = %d\n", object2->size());

	for (int i = 0; i < 2; i++) {		
		Hit hit;// = objects[i]->intersect(ray);

		if (hit.t > 0 && (firstHit.t < 0 || hit.t < firstHit.t)) {
			firstHit = hit;
		}
	}
	
	firstHit = object1->intersect(ray);
	Hit secondHit = object2->intersect(ray);
	if (secondHit.t > 0 && (firstHit.t < 0 || secondHit.t < firstHit.t)) {
		firstHit = secondHit;
	}
	



	if (dot(ray.dir, firstHit.normal) > 0) {
		firstHit.normal = (-1) * firstHit.normal;
	}

	if (firstHit.t >= 0.0f) {
		framebuffer[j * width + i] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, firstHit.normal);
	}

}
*/


/*
__global__ void renderFrame(Camera camera, Sphere object, vec3* frameBuffer, int width, int height) { // TODO: objects, lights, ezeket tárolhatjuk a hoston vectorban de a deviceon már arrayben kell lenniók, amiket ide adunk át

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	//printf("Kernel i: %d, j: %d\n", i, j);
	//printf("Camera pos: %.2f %.2f %.2f lookat: %.2f %.2f %.2f up: %.2f %.2f %.2f\n", camera.eye.x, camera.eye.y, camera.eye.z, camera.lookat.x, camera.lookat.y, camera.lookat.z, camera.up.x, camera.up.y, camera.up.z);
	//printf("Width: %d Height: %d\n", width, height);

	Hit firstHit;
	firstHit = object.intersect(ray);

	if (firstHit.t >= 0.0f) {
		if (dot(ray.dir, firstHit.normal) < 0) {
			firstHit.normal = firstHit.normal * (-1);
		}
		frameBuffer[j * width + i] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, firstHit.normal);
	}
}

__global__ void renderSpheres(Camera camera, Sphere* spheres, int num_spheres, float* ts, vec3* frameBuffer, int width, int height) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	Hit bestHit; bestHit.t = ts[j * width + i];

	//printf("Sphere center x: %.2f y: %.2f z: %.2f radius: %.2f\n", spheres[0].center.x, spheres[0].center.y, spheres[0].center.z, spheres[0].radius);

	bool newHitFound = false;

	for (int k = 0; k < num_spheres; k++) {
		Hit hit = spheres[k].intersect(ray);

		if (hit.t >= 0.0f && (bestHit.t < 0.0f || hit.t < bestHit.t)) {
			bestHit = hit;
			newHitFound = true;
		}
	}

	if (newHitFound) {
		ts[j * width + i] = bestHit.t;
		if (bestHit.t >= 0.0f) {
			if (dot(ray.dir, bestHit.normal) < 0) {
				bestHit.normal = bestHit.normal * (-1);
			}
			//printf("Hit with t: %.4f\n", bestHit.t);
			frameBuffer[j * width + i] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, bestHit.normal);
		}
		else {
			//frameBuffer[j * width + i] = vec3(0.5f, 0.5f, 0.5f) * dot(ray.dir, bestHit.normal);
		}
	}


}

__global__ void renderPlanes(Camera camera, Plane* planes, int num_planes, float* ts, vec3* frameBuffer, int width, int height) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	Hit bestHit; bestHit.t = ts[j * width + i];

	bool newHitFound = false;

	for (int k = 0; k < num_planes; k++) {
		Hit hit = planes[k].intersect(ray);

		if (hit.t >= 0.0f && (bestHit.t < 0.0f || hit.t < bestHit.t)) {
			bestHit = hit;
			newHitFound = true;
		}
	}

	if (newHitFound) {
		//bestHit.t >= 0.0f guaranteed
		ts[j * width + i] = bestHit.t;

		if (dot(ray.dir, bestHit.normal) < 0) {
			bestHit.normal = bestHit.normal * (-1);
		}
		frameBuffer[j * width + i] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, bestHit.normal);
	}
}

template<typename T>
__global__ void renderShapesTemplated(Camera camera, T* objects, int num_objects, Light* lights, int num_lights, float* ts, vec3* frameBuffer, int width, int height) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	int index = j * width + i;

	Hit bestHit; bestHit.t = ts[index];

	bool newHitFound = false;

	for (int k = 0; k < num_objects; k++) {
		Hit hit = objects[k].intersect(ray);

		if (hit.t >= 0.0f && (bestHit.t < 0.0f || hit.t < bestHit.t)) {
			bestHit = hit;
			newHitFound = true;
		}
	}

	if (newHitFound) {
		//bestHit.t >= 0.0f guaranteed
		ts[index] = bestHit.t;

		if (dot(ray.dir, bestHit.normal) < 0) {
			bestHit.normal = bestHit.normal * (-1);
		}

		for (int k = 0; k < num_lights; k++) {

		}

		frameBuffer[index] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, bestHit.normal);
	}
	else if (ts[index] < 0.0f) {
		//vec3(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f)
		frameBuffer[index] = vec3(0.521568f, 0.807843f, 0.921568f);
	}
}

__global__ void renderWithLights(Camera camera, ShapeData data, vec3* frameBuffer, int width, int height) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	Ray ray = camera.getRay(i, j);

	int index = j * width + i;

	Hit firstHit;

	for (int k = 0; k < data.num_spheres; k++) {
		Hit hit = data.spheres[k].intersect(ray);

		if (hit.t >= 0.0f && (firstHit.t < 0.0f || hit.t < firstHit.t)) {
			firstHit = hit;
		}
	}

	for (int k = 0; k < data.num_planes; k++) {
		Hit hit = data.planes[k].intersect(ray);

		if (hit.t >= 0.0f && (firstHit.t < 0.0f || hit.t < firstHit.t)) {
			firstHit = hit;
		}
	}


	if (firstHit.t >= 0.0f) {

		if (dot(ray.dir, firstHit.normal) < 0) {
			firstHit.normal = firstHit.normal * (-1);
		}

		for (int k = 0; k < data.num_lights; k++) {

		}
	}

	if (firstHit.t >= 0.0f) {
		frameBuffer[index] = vec3(1.0f, 1.0f, 1.0f) * dot(ray.dir, firstHit.normal);

	}
	else {
		//vec3(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f)
		frameBuffer[index] = vec3(0.521568f, 0.807843f, 0.921568f);
	}
}

*/
