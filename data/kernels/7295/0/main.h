#pragma once

// Simulation parameters
#define N_BODIES 100 // Number of bodies simulated
#define DURATION 5.0 // Duration for the simulation
#define DELTA_TIME 0.01 // duration of a single step
#define G_CONSTANT 4.302e-3 // Gravitational constant measured in - pc / M * (km/s)(km/s)
							// pc - parcec, M - solar mass unit
#define MASS_LOWER_BOUND 0.5 // Lower bound for randomly generated mass. 1 unit represents 1 solar unit
#define MASS_HIGHER_BOUND 2.5 // Lower bound for randomly generated mass. 1 unit represents 1 solar unit
#define POS_LOWER_BOUND -1.0 // Lowest posible x and/or y component for a starting position of the bodies. 1 unit represents 1 parcec
#define POS_HIGHER_BOUND 1.0 // Highest posible x and/or y component for a starting position of the bodies. 1 unit represents 1 parcec

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800

// Testing parameters
#define RUN_TESTS false // Whether to run testing or just generate a gif for demonstration
#define RANDOM_SEED 99 // Seed used to get pseudo-random numbers
#define NUMBER_OF_TESTS 20
#define BODY_INCREMENT 750 // How many bodies to add for each test
#define N_BODY_INCREMENTS 5 // How many times the number of bodies is incremented
#define MAX_THREAD_SHARING 125 // Maximum bodies calculated in single GPU thread
#define SHARING_MULTIPLIER 5 // More bodies per thread by this amount when testing
#define VALIDATION_OUTPUT true // Whether to output a gif and final positions or not