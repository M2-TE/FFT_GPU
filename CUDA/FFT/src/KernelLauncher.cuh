#pragma once

// no need to reinclude these
//#include <device_launch_parameters.h>
//#include <cuda_runtime.h>
//#include <stdio.h>
//#include <math.h>

template<int N, int M, int D>
__global__ void fft(float* A, float* ROT)
{
	//Declaration of arrays in shared memory.
	__shared__ float SA[M], SB[M], SROT[N];

	short tid = threadIdx.x;
	short n = logf(N) / logf(2);

	//loading "rotation elements" from Global memory to Shared memory with coalescence(in order of thread indices).
	SROT[tid] = ROT[tid];
	SROT[tid + D] = ROT[tid + D];

	//loading "input elements" from Global memory to Shared memory with coalescence(in order of thread indices).
	SA[tid] = A[tid];
	SA[tid + D] = A[tid + D];
	SA[tid + 2 * D] = A[tid + 2 * D];
	SA[tid + 3 * D] = A[tid + 3 * D];
	//synchronize all the threads untill all threads done their work(loading).
	__syncthreads();

	//FFT computations will be done inside for loop; one stage will be computed in each iteration. 
	for (short s = 1; s <= n; s++)
	{
		short p = (2 * N) / (1 << s);
		short ind0 = 2 * tid + (tid / (1 << (n - s))) * (1 << (n - s + 1));
		short ind1 = ind0 + p;

		//one butterfly operations will be computed in each thread. 
		//There are N/2 threads totally which performs N/2 butterfly operations in each stage. 
		//Input elements can be accessed in any order from Shared memory with out loss of performance.

		SB[ind0] = SA[ind0] + SA[ind1];//real parts addition.
		SB[ind0 + 1] = SA[ind0 + 1] + SA[ind1 + 1];//img parts subtraction.
		SB[ind1] = SA[ind0] - SA[ind1];//real parts subtraction.
		SB[ind1 + 1] = SA[ind0 + 1] - SA[ind1 + 1];//img parts subtraction.

		// "r" calculates which rotation element should be required(accessed) for particular thread.
		short r = (tid % (1 << (n - s))) * (1 << (s - 1));

		SA[ind0] = SB[ind0];
		SA[ind0 + 1] = SB[ind0 + 1];

		SA[ind1] = SB[ind1] * SROT[2 * r] + SB[ind1 + 1] * SROT[2 * r + 1];
		SA[ind1 + 1] = -SB[ind1] * SROT[2 * r + 1] + SB[ind1 + 1] * SROT[2 * r];

		//synchronize all the threads untill all threads done their work(FFT computations).
		__syncthreads();
	}

	//storing output elements from Shared memory to Global memory with coalescence(in order of thread indices).
	A[tid] = SA[tid];
	A[tid + D] = SA[tid + D];
	A[tid + 2 * D] = SA[tid + 2 * D];
	A[tid + 3 * D] = SA[tid + 3 * D];
}

template<int N>
void ExecuteFFT()
{
	static constexpr int nThreads = N / 2;
	static constexpr int nData = 2 * N; // size of input/output array
	static constexpr size_t dataWidth = sizeof(float) * nData;
	static constexpr size_t rotsWidth = sizeof(float) * N;

	float* pData = new float[N * 2];
	float* pRots = new float[N];
	float* pdData, *pdRots; // pointers to device ram (dram on GPU)

	//input elements of N-point FFT.
	for (int i = 0; i < N; i++)
	{
		pData[2 * i] = i;
		pData[2 * i + 1] = i;
	}

	//Rotations of N-point FFT.
	for (int j = 0; j < (N / 2); j++)
	{
		pRots[2 * j] = cosf((j * (6.2857)) / N);
		pRots[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	//Memory allocation in Global memory of Device(GPU).
	cudaMalloc((void**)&pdData, dataWidth);
	cudaMalloc((void**)&pdRots, rotsWidth);

	//Copying "input elements" and "rotations" of N-point FFT from CPU to GPU(global memory of GPU(Device)).
	cudaMemcpy(pdData, pData, dataWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(pdRots, pRots, rotsWidth, cudaMemcpyHostToDevice);

	dim3 gridDim(1);
	dim3 blockDim(nThreads);

	//kernel invocation
	fft<N, nData, nThreads> <<<gridDim, blockDim>>> (pdData, pdRots);

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(pData, pdData, dataWidth, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, pData[2 * l], 2 * l + 1, pData[2 * l + 1]);
}
