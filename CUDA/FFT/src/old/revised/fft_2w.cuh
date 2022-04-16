#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 64
#define M 2*N
#define D N/2

//GPU Part
__global__ void fft(float* A, float* ROT)
{
	//Declaration of arrays in shared memory.
	__shared__ float SA[M], /*SB[M],*/ SROT[N];

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

		float sb0 = SA[ind0] + SA[ind1];//real parts addition.
		float sb1 = SA[ind0 + 1] + SA[ind1 + 1];//img parts subtraction.

		float sb2 = SA[ind0] - SA[ind1];//real parts subtraction.
		float sb3 = SA[ind0 + 1] - SA[ind1 + 1];//img parts subtraction.

		// "r" calculates which rotation element should be required(accessed) for particular thread.
		short r = (tid % (1 << (n - s))) * (1 << (s - 1));

		SA[ind0] = sb0;
		SA[ind0 + 1] = sb1;

		SA[ind1] = sb2 * SROT[2 * r] + sb3 * SROT[2 * r + 1];
		SA[ind1 + 1] = -sb2 * SROT[2 * r + 1] + sb3 * SROT[2 * r];

		//synchronize all the threads untill all threads done their work(FFT computations).
		__syncthreads();
	}

	//storing output elements from Shared memory to Global memory with coalescence(in order of thread indices).
	A[tid] = SA[tid];
	A[tid + D] = SA[tid + D];
	A[tid + 2 * D] = SA[tid + 2 * D];
	A[tid + 3 * D] = SA[tid + 3 * D];
}

// CPU Part
void Do()
{
	float A[2 * N];
	float* Ad;
	float ROT[N];
	float* ROTd;

	int memsize = 2 * N * sizeof(float);
	int rotsize = N * sizeof(float);

	//input elements of N-point FFT.
	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}

	//Rotations of N-point FFT.
	for (int j = 0; j < (N / 2); j++)
	{
		ROT[2 * j] = cosf((j * (6.2857)) / N);
		ROT[2 * j + 1] = sinf((j * (6.2857)) / N);
	}

	//Memory allocation in Global memory of Device(GPU).
	cudaMalloc((void**)&Ad, memsize); //for input elements.
	cudaMalloc((void**)&ROTd, rotsize); //for rotations.

		//Copying "input elements" and "rotations" of N-point FFT from CPU to GPU(global memory of GPU(Device)).
	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(ROTd, ROT, rotsize, cudaMemcpyHostToDevice);

	dim3 gridDim(1, 1);
	dim3 blockDim(D, 1);

	//kernel invocation. 
	//Func<<< Dg, Db, Ns  >>>(parameter);
	fft << <gridDim, blockDim >> > (Ad, ROTd);

	//Copy output elements from Device to CPU after kernel execution.
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);

	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++)
		printf("RE:A[%d]=%f\t\t\t, IM: A[%d]=%f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);

}
