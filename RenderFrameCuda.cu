#pragma once
#include "RenderFrameCuda.cuh"

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

__global__ void renderGPU(Camera camera, Intersectable*** objects, int num_objects, Light* lights, int num_lights, vec3* frameBuffer, int width, int height) {
	//if the coordinates are out of bounds, return
	if ((blockIdx.x * blockDim.x) + threadIdx.x > width || (blockIdx.y * blockDim.y) + threadIdx.y > height) return;

	//epsilon used for comparing floating-point numbers
	float epsilon = 1e-4;

	//number of reflected rays to compute
	int depth = 4;

	//pixel positions
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	//ray to compute
	Ray ray = camera.getRay(i, j);

	//positions of pixel in the one dimensional framebuffer
	int index = j * width + i;

	Hit firstHit = getFirstHit(objects, num_objects, ray);

	if (firstHit.t >= 0.0f) {

		//ambient
		vec3 color = firstHit.material->ka;

		for (int k = 0; k < num_lights; k++) {
			Light& light = lights[k];

			//directional light
			if (light.isDirectional) {
				Ray lightRay = Ray(firstHit.position + (-light.pos_dir) * epsilon, -light.pos_dir);

				if (!DirectionalLightShadowed(objects, num_objects, lightRay)) {
					//diffuse
					color = color + light.color * firstHit.material->kd * dot(light.pos_dir, firstHit.normal);

					//Blinn-Phong specular
					vec3 halfway = normalize(lightRay.dir + ray.dir);
					float cosDelta = dot(halfway, firstHit.normal);

					if (cosDelta > 0.0f) {
						color = color + light.color * firstHit.material->ks * powf(cosDelta, firstHit.material->shininess);
					}
				}
			}

			//point light
			else {
				Ray lightRay = Ray(light.pos_dir, normalize(firstHit.position - light.pos_dir));

				Hit lightHit = getFirstHit(objects, num_objects, lightRay);

				if (lightHit.t > 0.0f && vec3Equal(firstHit.position, lightHit.position)) {

					float distance = 1.0f + length(light.pos_dir - lightHit.position);

					float intensity = 1.0f / (distance * distance);

					//diffuse
					color = color + light.color * firstHit.material->kd * dot(lightHit.normal, lightRay.dir) * intensity;

					//Blinn-Phong specular
					vec3 halfway = normalize(lightRay.dir + ray.dir);
					float cosDelta = dot(halfway, firstHit.normal);

					if (cosDelta > 0.0f) {
						color = color + light.color * firstHit.material->ks * powf(cosDelta, firstHit.material->shininess) * intensity;
					}
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
