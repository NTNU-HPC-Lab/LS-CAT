#include "includes.h"
__global__ void apply_lifter_and_floor_energy( int num_frames, int num_cols, float cepstral_lifter, bool use_energy, float energy_floor, float *log_energy, float *lifter_coeffs, float *features, int32_t ldf) {
int thread_id = threadIdx.x;
int frame = blockIdx.x;

float *feats = features + frame * ldf;

// apply lifter coefficients
if (cepstral_lifter != 0.0f) {
for (int c = thread_id; c < num_cols; c += CU1DBLOCK) {
float lift = lifter_coeffs[c];
float f = feats[c];
feats[c] = f * lift;
}
}

// Thread 0 for each frame will apply energy
if (use_energy && thread_id == 0) {
float energy = log_energy[frame];
float log_energy_floor = log(energy_floor);

if (energy_floor > 0.0f && energy < log_energy_floor) {
energy = log_energy_floor;
}
feats[0] = energy;
}
}