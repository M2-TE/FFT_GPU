#pragma once

#include "custom_fft_kernels.cuh"

void perform_custom_fft(FFTData& data)
{
	dim3 gridDim(1, 1, 1);
	dim3 blockDim(8, data.size() / 64, 1);

	fft KERNEL_GRID(gridDim, blockDim)(data.deviceData, data.deviceData);
}