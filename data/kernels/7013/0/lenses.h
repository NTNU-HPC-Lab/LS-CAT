#ifndef LENSES_H
#define LENSES_H

// Implement lens equation, given the lens position (xl, yl) and the
// lens system configuration, shoot a ray back to the source position
// (xs, ys)
void shoot(float& xs, float& ys, float xl, float yl, 
	   float* xlens, float* ylens, float* eps, int nlenses);

// Set up a single lens example
int set_example_1(float** xlens, float** ylens, float** eps);

// Simple binary lens
int set_example_2(float** xlens, float** ylens, float** eps);

// Triple lens
int set_example_3(float** xlens, float** ylens, float** eps);

float pick_random(float x1, float x2);

// Many lenses
int set_example_n(const int nuse, float** xlens, float** ylens, float** eps);

#endif
