#pragma once

#include <vector>

class Par_handler
{
public:
	Par_handler(int n_of_bodies);
	~Par_handler();
	void physics_step(std::vector<double> &bodies_formated, const double &dt, const uint16_t &bods_per_thread);
private:
	double *before, *after;
	unsigned long size;
};