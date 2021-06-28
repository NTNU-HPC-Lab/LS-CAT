#ifndef PARTICLES_H_
#define PARTICLES_H_

#include "grid.cuh"
#include "helpers.cuh"

struct Particle{
    //keeps information about the position of one particle in (6D) phase space (positions, velocities)
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
};

struct Species{
    //keeps information about one distinct group of particles
    float m; //mass
    float q; //charge

    //number of particles in group: not fully used yet
    long int N_particles_1_axis;
    long int N_particles;
    long int N;

    Particle *particles;
    Particle *d_particles;

    dim3 particleThreads;
    dim3 particleBlocks;

    float *d_block_v2s;
    float *block_v2s;

    float *d_block_Px;
    float *block_Px;

    float *d_block_Py;
    float *block_Py;

    float *d_block_Pz;
    float *block_Pz;

    float *d_sum_Px;
    float *d_sum_Py;
    float *d_sum_Pz;
    float *d_sum_v2s;

    float *moments;
    float *d_moments;

    float KE;
    float Px;
    float Py;
    float Pz;

    // Particle total_values;
    // float total_vabs;
    // float T;
    // float kinetic_E;
    // float potential_E;
};

void init_species(Species *s, float shiftx, float shifty, float shiftz,
    float vx, float vy, float vz,
    int N_particles_1_axis, int N_grid, float dx);
void dump_position_data(Species *s, char* name);
void scatter_charge(Species *s, Grid*g);
void InitialVelocityStep(Species *s, Grid *g, float dt);
void SpeciesPush(Species *s, Grid *g, float dt);
void particle_cleanup(Species *s);

#endif
