#pragma once
#include <cstdint>
#include <memory>
using std::unique_ptr;

struct Lab2VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab2VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab2VideoGenerator();
	~Lab2VideoGenerator();
	void get_info(Lab2VideoInfo &info);
    void init();
    void UpdateAccelerateField();
    void UpdatePressureField();
    void UpdateVelocityField();
    void UpdateDensityField(bool isDrip, int cX, int cY, int r);
	void Generate(uint8_t *yuv);

private:
    float *d_curr, *d_last;
    float *u_dimX, *u_dimY;
    float *w_dimX, *w_dimY;
    float *f_dimX, *f_dimY;
    float *p;
};
