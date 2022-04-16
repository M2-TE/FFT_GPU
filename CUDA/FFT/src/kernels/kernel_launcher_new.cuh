#pragma once

#include "kernel_fft_new.cuh"

template<int nInput>
float ExecuteFFTNew(bool bPrintOutput = false, uint nRepetitions = 1u)
{
	static constexpr int N = nInput;
	static constexpr int nBlocks = 1;
	static constexpr int nThreads = N / 2;
	static constexpr int nData = 2 * N; // size of input/output array
	static constexpr uint nStages = gcem::log(static_cast<float>(nInput)) / gcem::log(2.0f); // word group size of 2

	static constexpr size_t dataWidth = sizeof(float) * nData;
	static constexpr size_t rotsWidth = sizeof(float) * N;
	static constexpr size_t cyclesWidth = sizeof(uint) * nThreads;

	// pointers to ram
	float* pData = new float[N * 2];
	float* pRots = new float[N];
	uint* pCycles = new uint[nThreads];

	// pointers to device ram (dram on GPU)
	float* pdData, *pdRots; 
	uint* pdCycles;

	//input elements of N-point FFT.
	for (uint i = 0; i < N; i++) {
		pData[i] = i; // real
		pData[i + N] = i; // imaginary
	}

	// TODO: fix the horrible loose type usage in these two loops (float/double/int/etc)
	//Rotations of N-point FFT.
	for (int j = 0; j < (N / 2); j++)
	{
		pRots[2 * j] = cosf((j * (6.2857)) / N); // isnt 6.whatever just PI * 2?
		pRots[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	//Memory allocation in Global memory of Device(GPU).
	cudaMalloc(reinterpret_cast<void**>(&pdData), dataWidth);
	cudaMalloc(reinterpret_cast<void**>(&pdRots), rotsWidth);
	cudaMalloc(reinterpret_cast<void**>(&pdCycles), cyclesWidth);

	//Copying "input elements" and "rotations" of N-point FFT from CPU to GPU(global memory of GPU(Device)).
	cudaMemcpy(pdData, pData, dataWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(pdRots, pRots, rotsWidth, cudaMemcpyHostToDevice);

	//kernel invocation
	CUDA_TIMER_START();
	for (uint i = 0u; i < nRepetitions; i++) {
		KernelFFTNew<nInput, nData, 2, nThreads, nStages>KERNEL_GRID(nBlocks, nThreads)(pdData, pdRots, pdCycles);
		cudaDeviceSynchronize();
	}
	CUDA_TIMER_END();

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(pData, pdData, dataWidth, cudaMemcpyDeviceToHost);
	cudaMemcpy(pCycles, pdCycles, cyclesWidth, cudaMemcpyDeviceToHost);

	if (bPrintOutput) {
		printf("The  outputs are: \n");
		for (int l = 0; l < N; l++) {
			//printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", l, pData[l], l + N, pData[l + N]);
		}

		// min/max/avg cycles
		uint min = 1000000u, max = 0u;
		float avg = 0.0f;
		for (int i = 0; i < nThreads; i++) {
			uint cycles = pCycles[i];
			min = cycles < min ? cycles : min;
			max = cycles > max ? cycles : max;
			avg += static_cast<float>(cycles);
		}
		avg /= static_cast<float>(nThreads);
		printf("Cycles: min %d, max %d, avg %.2f\n", min, max, avg);
		printf("Time for the kernel: %f us\n", time * 1000.0);
	}

	// clean up
	delete pData, pRots, pCycles;
	cudaFree(pdData);
	cudaFree(pdRots);
	cudaFree(pdCycles);
	return time;
}
