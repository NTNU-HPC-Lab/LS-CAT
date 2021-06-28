#pragma once


namespace proto
{

extern void findMedian(const ip::CpuImageU8& image, double* median);
extern void findMedian(const ip::CpuImageU16& image, double* median);

extern void findMedian(const ip::GpuImageU8& image, double* devMedian);
extern void findMedian(const ip::GpuImageU16& image, double* devMedian);


}
