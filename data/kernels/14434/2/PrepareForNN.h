#ifndef PREPAREFORNN_H
#define PREPAREFORNN_H

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

extern float * g_sweepers_d, *g_mines_d, *g_distances_d, *g_inputs_d;
extern int * g_scores_d, * g_mineIdx_d;

void set_up_prepare_for_NN(int num_mines, int num_sweepers);

void end_prepare_for_NN();

//takes in sweeper positions and mine positions, sets inputs vector to prepare for neural network kernel
void call_cuda_prepare_for_NN(float * sweeper_pos_v, float * mine_pos_v, float * inputs, int * sweeper_score_v, int num_sweeprs, int num_mines, int width, int height, int size);

//sets distance v to the distance from every sweeper to every mine
__global__ void calculate_distances(float * sweeper_pos_v, float * mine_pos_v, int num_sweepers, int num_mines, float * distance_v, float * inputs, int * sweeper_score_v, int width, int height, int size);

//takes the distance vector and uses a reduction to find the minimum one per sweeper
//threads: dim3(32,1,1) blocks: dim3(ciel(num_mines / 64), num_sweepers,1)
// over 64 because half as many threads are required to reduce the vector
//each row is one reduction of one sweeper and all reductions are done at once
//then result is placed into inputs
//CHANGES SWEEPER_POS_V
__global__ void find_closest_mine(float * mine_pos_v, float * distances_v, int * mineIdx_v, int num_sweeprs, int num_mines, float * inputs);

#endif //PREPAREFORNN_H