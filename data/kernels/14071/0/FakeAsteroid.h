/*
 * FakeAsteroid.h
 *
 *  Created on: Nov 13, 2016
 *      Author: peter
 */

#ifndef FAKEASTEROID_H_
#define FAKEASTEROID_H_

#include <random>
#include <omp.h>
#include <fitsio.h>
#include "GeneratorPSF.h"

class FakeAsteroid
{
	public:
		FakeAsteroid();
		void createImage(short *image, int width, int height,
			float xpos, float ypox, psfMatrix psf, float asteroidLevel, float noiseLevel);
};

#endif /* FAKEASTEROID_H_ */
