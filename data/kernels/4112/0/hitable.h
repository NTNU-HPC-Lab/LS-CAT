#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

struct hit_record
{
	float t;
	float3 p;
	float3 normal;
	bool isLight;
};

class hitable {
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif