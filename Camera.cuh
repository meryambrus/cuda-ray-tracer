#pragma once

#include "Ray.cuh"

extern const int WINDOW_WIDTH;
extern const int WINDOW_HEIGHT;

enum class MoveDirection {
    FORWARD,
    LEFT,
    RIGHT,
    BACKWARD
};

class Camera {
public:
    vec3 eye, lookat, right, up;
    int width, height;
    float fov;
    float aspectRatio;
public:
    __host__ void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov);

    __host__ __device__ Ray getRay(int X, int Y);

    __host__ void move(float distance, MoveDirection dir);

    __host__ void rotate(float angle_rad, vec3 axis);
};
