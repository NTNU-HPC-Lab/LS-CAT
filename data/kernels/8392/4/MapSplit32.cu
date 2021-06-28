#include "includes.h"
#pragma once

/*
//åñëè äëÿ âñåõ êàðò õâàòèò âîçìîæíîñòåé âèäåîêàðòû (ÐÀÁÎÒÀÅÒ)
*/






__global__ void MapSplit32(const int* one, int* result, unsigned int mx, unsigned int width)
{
unsigned int ppp = blockIdx.x * blockDim.x * 32 + threadIdx.x;
unsigned int rix = ppp % width;
unsigned int riy = (ppp / mx) + ((ppp % mx) / width);
unsigned int xxx = riy * width + rix;
unsigned int ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];

ppp++;
rix = ppp % width;
riy = (ppp / mx) + ((ppp % mx) / width);
xxx = riy * width + rix;
ddx = riy * mx + rix;
result[xxx] = one[ddx];
}