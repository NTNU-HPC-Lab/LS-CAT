#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

class Camera {
    public:
        __device__ Camera() {
            lower_left_corner = Vector3(-2.0, -1.0, -1.0);
            horizontal = Vector3(4.0, 0.0, 0.0);
            vertical = Vector3(0.0, 2.0, 0.0);
            origin = Vector3(0.0, 0.0, 0.0);
        }

        __device__ Ray get_Ray(float u, float v) { return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }
        Vector3 origin;
        Vector3 lower_left_corner;
        Vector3 horizontal;
        Vector3 vertical;
};

#endif