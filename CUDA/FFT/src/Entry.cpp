#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// constexpr math library
#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "kernels/kernel_launcher.cuh"
#include "kernels/kernel_launcher_new.cuh"
#include "cufft/cufft_impl.cuh"


int main()
{
	static constexpr uint N = 4; // N-point fft
	//ExecuteFFT<N>(1u, true);
	//system("pause");
	//ExecuteFFTNew<N>(false);
	DoCUFFT(N, 1u, true);

	//printf("Res: %d\n", 0b0010 >> 1);

	// UNRELIABLE
	// performance comparison (Original VS cuFFT)
	//if (false) {
	//	// 2048 is the limit (thread limit per SM)
	//	uint n = 1u, rep = 100u;
	//	float a = 0.0f, b = 0.0f;
	//	for (uint i = 0u; i < n; i++) a += ExecuteFFT<N>(rep);
	//	for (uint i = 0u; i < n; i++) b += DoCUFFT(N, rep); // reusing an already existing plan results in higher throughput
	//	printf("Results a: %5.5f\t\t\t Results b: %5.5f", a, b);
	//}

}