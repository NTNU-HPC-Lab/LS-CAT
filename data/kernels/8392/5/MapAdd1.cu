#include "includes.h"
#pragma once

/*
//åñëè äëÿ âñåõ êàðò õâàòèò âîçìîæíîñòåé âèäåîêàðòû (ÐÀÁÎÒÀÅÒ)
*/






__global__ void MapAdd1(int* one, const int* result, unsigned int mx, unsigned int width)
{
const unsigned int ppp = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int rix = ppp % width;
const unsigned int riy = (ppp / mx) + ((ppp % mx) / width);
const unsigned int xxx = riy * width + rix;
const unsigned int ddx = riy * mx + rix;
one[ddx] = result[xxx];
}