#ifndef FLOATVOLUME_H
#define FLOATVOLUME_H

// index = y * width * depth + x * depth + z
// given a cube with one face towards you,
// index 0 is at top left, closest to you
struct FloatVolume {
  unsigned width, height, depth;
  float *contents;
};

void setEmptyFloatVolume(struct FloatVolume *fv, unsigned width,
  unsigned height, unsigned depth);

void printFloatVolume(struct FloatVolume *fv);

#endif
