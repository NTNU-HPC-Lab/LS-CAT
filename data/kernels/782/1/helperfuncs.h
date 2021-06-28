#ifndef HELPERFUNCS_H
#define HELPERFUNCS_H

struct FloatVolume;

// y-coordinate = a, x-coordinate = b
unsigned toIndex2D(unsigned a, unsigned b, unsigned blen);

// y-coordinate = a, x-coordinate = b, z-cooridnate = c
unsigned toIndex3D(unsigned a, unsigned b, unsigned blen, unsigned c,
  unsigned clen);

// Given two colors, determine difference
float diffColor(unsigned char *c1, unsigned char *c2);

// Return 1 if difference, 0 if none
int compareFloatVolumes(struct FloatVolume *fv1, struct FloatVolume *fv2);

#endif
