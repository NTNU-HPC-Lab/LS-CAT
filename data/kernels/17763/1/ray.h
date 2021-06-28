#ifndef RAYH
#define RAYH
#include "vector3.h"

class Ray
{
    public:
        __device__ Ray() {}
        __device__ Ray(const Vector3& a, const Vector3& b) { A = a; B = b; }
        __device__ Vector3 origin() const       { return A; }
        __device__ Vector3 direction() const    { return B; }
        __device__ Vector3 point_at_parameter(float t) const { return A + t*B; }
        Vector3 A;
        Vector3 B;
};

#endif