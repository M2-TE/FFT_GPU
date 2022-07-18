#pragma once

#include "custom_fft_kernels.cuh"

// TODO: circumvent need for template, use switch to select required kernel
template<uint N> inline void perform_custom_fft(FFTData<float>& data)
{
	uint nX = std::min<uint>(std::max<uint>(N / 8u, 1u), 8u);
	uint nY = std::max<uint>(N / 64u, 1u);
	dim3 gridDim(1, 1, 1);
	dim3 blockDim(nX, nY, 1);
	printf("Launched kernel with x: %d, y: %d\n", blockDim.x, blockDim.y);

	Timer timer;
	fft<N> KERNEL_GRID(gridDim, blockDim)(data.deviceData, data.deviceData);
	timer.eval_print();
}