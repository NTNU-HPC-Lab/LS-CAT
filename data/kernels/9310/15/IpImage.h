#pragma once

#include "IpStruct.h"
#include "IpImageCPU.h"
#include "IpImageGPU.h"

namespace ip
{

void printImage(const ip::CpuImageU8&  img);
void printImage(const ip::CpuImageU16& img);
void printImage(const ip::CpuImageU32& img);
void printImage(const ip::CpuImageF32& img);
void printImage(const ip::CpuImageF64& img);

void printImage(const ip::GpuImageU8 & img);
void printImage(const ip::GpuImageU16& img);
void printImage(const ip::GpuImageU32& img);
void printImage(const ip::GpuImageF32& img);
void printImage(const ip::GpuImageF64& img);

} // namespace ip

