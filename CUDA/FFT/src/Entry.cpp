#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// constexpr math library (additional include dir doesnt seem to work lol)
#include "../vendor/gcem/include/gcem.hpp"

// cuFFT
#include <cufft.h>

// src code
#include "Utils.cuh"
#include "KernelLauncher.cuh"
#include "CufftImpl.cuh"
//#include "revised/fft_2w.cuh"
//#include "revised/fft_cg_2w.cuh"
//#include "revised/fft_mk_cg_r2_2_1024.cuh"


int main()
{
	static constexpr uint N = 32; // N-point fft
	ExecuteFFT<N>(1u);
	DoCUFFT(N, 1u);

	// 2048 is the limit (thread limit per SM)
	float a = 0.0f, b = 0.0f;
	uint n = 1000u, rep = 1u;
	for (uint i = 0u; i < n; i++) a += ExecuteFFT<N>(rep);
	for (uint i = 0u; i < n; i++) b += DoCUFFT(N, rep); // reusing an already existing plan results in higher throughput

	printf("Results a: %5.5f\t\t\t Results b: %5.5f", a, b);
}