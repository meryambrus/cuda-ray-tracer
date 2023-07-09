#pragma once
#include "Material.cuh"

__host__ __device__ Material::Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) {
	ka = _ka; kd = _kd; ks = _ks; shininess = _shininess;
}

__host__ __device__ Material::Material(vec3 objectColor, float _shininess)
{
	ka = objectColor * 0.2f;
	kd = objectColor * 0.8f;
	ks = objectColor;
	shininess = _shininess;
}

__host__ __device__ Material::Material(const Material& obj)
{
	this->ka = obj.ka;
	this->kd = obj.kd;
	this->ks = obj.ks;
	this->shininess = obj.shininess;
}

__host__ __device__ void Material::operator=(const Material& obj)
{
	this->ka = obj.ka;
	this->kd = obj.kd;
	this->ks = obj.ks;
	this->shininess = obj.shininess;
}
