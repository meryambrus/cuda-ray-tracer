#pragma once
#include "Light.cuh"

__host__ __device__ Light::Light(vec3 _pos_dir, vec3 _color, LightType _type) {
	pos_dir = _pos_dir; color = _color;
	isDirectional = _type == LightType::DIRECTIONAL;
}
