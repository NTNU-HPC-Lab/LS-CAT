#ifndef PATHVOLUME_H
#define PATHVOLUME_H

struct FloatVolume;

// Fill cells of path volume where x = 0 and y = 0
void setX0Y0(struct FloatVolume *pv, struct FloatVolume *dv);

// Fill cells where z = 0 and y = 0
void setZ0Y0(struct FloatVolume *pv, struct FloatVolume *dv);

// Fill cells where z = 0 and x = 0
void setZ0X0(struct FloatVolume *pv, struct FloatVolume *dv);

// Fill cells where x = 0, assuming Z0X0 and Z0Y0 are filled
void setX0(struct FloatVolume *pv, struct FloatVolume *dv);

// Fill cells where y = 0, assuming Z0Y0 and X0Y0 are filled
void setY0(struct FloatVolume *pv, struct FloatVolume *dv);

// Fill cells where z = 0, assuming Z0Y0 and Z0X0 are filled
void setZ0(struct FloatVolume *pv, struct FloatVolume *dv);

// Function too thicc
void pathVolumeInit(struct FloatVolume *pv, struct FloatVolume *dv);

#endif
