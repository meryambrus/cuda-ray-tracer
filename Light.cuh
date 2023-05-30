#pragma once

#include "framework.h"

enum class LightType {
	POINT, DIRECTIONAL
};

class Light {
public:
	vec3 pos_dir; //pos if point, dir id directional
	vec3 color;
	bool isDirectional;
	__host__ __device__ Light(vec3 _pos_dir, vec3 _color, LightType _type);
};
