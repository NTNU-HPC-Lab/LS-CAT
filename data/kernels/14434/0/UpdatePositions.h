#ifndef UPDATEPOSITIONS_H
#define UPDATEPOSITIONS_H

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

extern float * g_outputs_d, * g_sweepers_d_2;

void set_up_update_positions(int num_sweepers);

void end_update_positions();

void call_cuda_update_positions(int num_sweepers, float max_speed, float * outputs, float * sweepers);

__global__ void update_positions(float max_speed, float * outputs_d, float * sweepers_d);

#endif //UPDATEPOSITIONS_H