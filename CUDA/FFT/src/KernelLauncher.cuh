#pragma once

#include "KernelFFT.cuh"

template<int nInput>
void ExecuteFFT()
{
	static constexpr int N = nInput;
	static constexpr int nBlocks = 1;
	static constexpr int nThreads = N / 2;
	static constexpr int nData = 2 * N; // size of input/output array
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

	// TODO: fix the horrible loose type usage in these two loops (float/double/int/etc)
	//input elements of N-point FFT.
	for (int i = 0; i < N; i++)
	{
		pData[2 * i] = i;
		pData[2 * i + 1] = i;
	}

	//Rotations of N-point FFT.
	for (int j = 0; j < (N / 2); j++)
	{
		pRots[2 * j] = cosf((j * (6.2857)) / N); // isnt 6.whatever just PI * 2?
		pRots[2 * j + 1] = sinf((j * (6.2857)) / N);
	}
	for (int i = 0; i < nThreads; i++) pCycles[i] = 0u; // unnecessary

	//Memory allocation in Global memory of Device(GPU).
	cudaMalloc(reinterpret_cast<void**>(&pdData), dataWidth);
	cudaMalloc(reinterpret_cast<void**>(&pdRots), rotsWidth);
	cudaMalloc(reinterpret_cast<void**>(&pdCycles), cyclesWidth);

	//Copying "input elements" and "rotations" of N-point FFT from CPU to GPU(global memory of GPU(Device)).
	cudaMemcpy(pdData, pData, dataWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(pdRots, pRots, rotsWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(pdCycles, pCycles, cyclesWidth, cudaMemcpyHostToDevice); // unnecessary

	//kernel invocation
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	KernelFFT<nInput, nData, 2, nThreads, 10>KERNEL_GRID(nBlocks, nThreads)(pdData, pdRots, pdCycles);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(pData, pdData, dataWidth, cudaMemcpyDeviceToHost);
	cudaMemcpy(pCycles, pdCycles, cyclesWidth, cudaMemcpyDeviceToHost);

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

	//printf("The  outputs are: \n");
	//for (int l = 0; l < N; l++) printf("RE:A[%d]=%.2f\t\t\t, IM: A[%d]=%.2f\t\t\t \n ", 2 * l, pData[2 * l], 2 * l + 1, pData[2 * l + 1]);

	// clean up
	delete pData, pRots, pCycles;
}
