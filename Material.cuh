#pragma once

#include "framework.h"



class Material {
public:
	vec3 ka, kd, ks; //ambient, diffuse, specular color
	float shininess;

	__host__ __device__ Material(vec3 _ka, vec3 _kd, vec3 _ks = vec3(1.0f, 1.0f, 1.0f), float _shininess = 32.0f);

	__host__ __device__ Material(vec3 objectColor, float _shininess = 32.0f);

	__host__ __device__ Material(const Material& obj);

	__host__ __device__ void operator=(const Material& obj);
};