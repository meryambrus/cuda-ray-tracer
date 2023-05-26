#pragma once

#include "Sphere.cuh"
#include "Plane.cuh"
#include "Light.cuh"

struct ShapeData {
	Sphere* spheres;	int num_spheres;
	Plane* planes;		int num_planes;
	Light* lights;		int num_lights;

	ShapeData(Sphere* _spheres, int _num_spheres, Plane* _planes, int _num_planes, Light* _lights, int _num_lights) {
		spheres = _spheres;		num_spheres = _num_spheres;
		planes = _planes;		num_planes = _num_planes;
		lights = _lights;		num_lights = _num_lights;
	}
};