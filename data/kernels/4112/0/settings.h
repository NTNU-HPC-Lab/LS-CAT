#ifndef SETTINGSH
#define SETTINGSH

#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cout << "CUDA error" << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

__device__ float3 random_direction(curandState *local_rand_state) {
	float3 p;
	do {
		p = 2.0*make_float3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
			- make_float3(1.0, 1.0, 1.0);
	} while (squared_length(p) >= 3.0);
	return p;
}

class camera {
public:
	__device__ camera() {
		lower_left_corner = make_float3(-2.0, -1.0, -1.0);
		horizontal = make_float3(4.0, 0.0, 0.0);
		vertical = make_float3(0.0, 2.0, 0.0);
		origin = make_float3(1, 0.0, 2.0);
	}
	__device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); }

	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
};

#endif