#pragma once
#include "Camera.cuh"

__host__ void Camera::set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
    eye = _eye;
    lookat = _lookat;
    fov = _fov;
    vec3 w = eye - lookat;
    float focus = length(w);
    right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
    up = normalize(cross(w, right)) * focus * tanf(fov / 2);

    width = WINDOW_WIDTH; height = WINDOW_HEIGHT; aspectRatio = (float)width / (float)height;
}

__host__ __device__ Ray Camera::getRay(int X, int Y) {
    vec3 dir = lookat + (right * (2.0f * (X + 0.5f) / width - 1) * aspectRatio) + up * (2.0f * (Y + 0.5f) / height - 1) - eye;
    return Ray(eye, dir);
}

__host__ void Camera::move(float distance, MoveDirection dir) {
    vec3 lookdir = normalize(lookat - eye);
    vec3 right_dir = cross(lookdir, normalize(up));
    switch (dir) {
    case MoveDirection::FORWARD:
        eye = eye + lookdir * distance;
        lookat = lookat + lookdir * distance;
        break;
    case MoveDirection::BACKWARD:
        eye = eye - lookdir * distance;
        lookat = lookat - lookdir * distance;
        break;
    case MoveDirection::LEFT:
        eye = eye - right_dir * distance;
        lookat = lookat - right_dir * distance;
        break;
    case MoveDirection::RIGHT:
        eye = eye + right_dir * distance;
        lookat = lookat + right_dir * distance;
        break;
    }
}

__host__ void Camera::rotate(float angle_rad, vec3 axis) {
    mat4 rotmat = RotationMatrix(angle_rad, axis);

    vec4 lookat_dir = vec4(lookat - eye, 1.0f);

    vec4 result = lookat_dir * rotmat;

    lookat = eye + normalize(vec3(result.x, result.y, result.z));
    set(eye, lookat, up, fov);
}