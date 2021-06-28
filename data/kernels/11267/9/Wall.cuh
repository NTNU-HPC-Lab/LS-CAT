#ifndef WALL_CUH
#define WALL_CUH
#include "Float3.cuh"
#include <glm/vec3.hpp>

using namespace glm;

class Wall
{
public:
	__host__ __device__ Wall();
	__host__ __device__ Wall(vec3 _min, vec3 _max);
	vec3 max;
	vec3 min;
};

#endif