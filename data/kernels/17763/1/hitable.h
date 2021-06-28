#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

struct hit_record
{
    float t;
    Vector3 p;
    Vector3 normal;
};

class Hitable  {
    public:
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif