#pragma once

#include "framework.h"

struct Material {
	vec3 ka, kd, ks; //ambient, diffuse, specular color
	float shininess;
	float roughness;

	Material(vec3 _ka, vec3 _kd, float _shininess, float _roughness);
};