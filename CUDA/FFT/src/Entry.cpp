#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// constexpr math library
#include "gcem/include/gcem.hpp"

// cuFFT
#include <cufft.h>

// src code
#include "utils.cuh"
#include "kernels/kernel_launcher.cuh"
#include "kernels/kernel_launcher_new.cuh"
#include "cufft/cufft_impl.cuh"
//#include "old/revised/fft_2w.cuh"
//#include "old/revised/fft_cg_2w.cuh"
//#include "old/revised/fft_mk_cg_r2_2_1024.cuh"


int main()
{
	static constexpr uint N = 2048; // N-point fft
	ExecuteFFTNew<N>(true);
	ExecuteFFT<N>(1u, true);
	//DoCUFFT(N, 1u);
	
	//// performance comparison (Original VS New)
	//if (true) {
	//	// 2048 is the limit (thread limit per SM)
	//	uint n = 100u, rep = 1u;
	//	float a = 0.0f, b = 0.0f;
	//	for (uint i = 0u; i < n; i++) a += ExecuteFFT<N>(rep);
	//	for (uint i = 0u; i < n; i++) b += ExecuteFFTNew<N>(); // reusing an already existing plan results in higher throughput
	//	printf("Results a: %5.5f\t\t\t Results b: %5.5f", a, b);
	//}

	// UNRELIABLE
	// performance comparison (Original VS cuFFT)
	if (false) {
		// 2048 is the limit (thread limit per SM)
		uint n = 100u, rep = 1u;
		float a = 0.0f, b = 0.0f;
		for (uint i = 0u; i < n; i++) a += ExecuteFFT<N>(rep);
		for (uint i = 0u; i < n; i++) b += DoCUFFT(N, rep); // reusing an already existing plan results in higher throughput
		printf("Results a: %5.5f\t\t\t Results b: %5.5f", a, b);
	}
}