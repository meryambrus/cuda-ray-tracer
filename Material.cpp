#pragma once
#include "Material.h"

Material::Material(vec3 _ka, vec3 _kd, float _shininess, float _roughness) {
	ka = _ka; kd = _kd; shininess = _shininess; roughness = _roughness;
}