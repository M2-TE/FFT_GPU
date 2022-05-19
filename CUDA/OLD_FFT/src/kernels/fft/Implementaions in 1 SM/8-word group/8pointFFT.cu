#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ void execute_8point_fft(float* A)
{
	const unsigned int tid = threadIdx.x;
	const float coef = sqrtf(2.0f) / 2.0f;
	const unsigned int N = 8;
	const unsigned int step = N * 2;

	// stage 1
	// butterflies
	float x0 = A[tid * step + 0] + A[tid * step + 8]; // R
	float x1 = A[tid * step + 1] + A[tid * step + 9]; // I
	float x8 = A[tid * step + 0] - A[tid * step + 8]; // R
	float x9 = A[tid * step + 1] - A[tid * step + 9]; // I

	float x2 = A[tid * step + 2] + A[tid * step + 10]; // R
	float x3 = A[tid * step + 3] + A[tid * step + 11]; // I
	float x10 = A[tid * step + 2] - A[tid * step + 10]; // R
	float x11 = A[tid * step + 3] - A[tid * step + 11]; // I

	float x4 = A[tid * step + 4] + A[tid * step + 12]; // R
	float x5 = A[tid * step + 5] + A[tid * step + 13]; // I
	float x12 = A[tid * step + 5] - A[tid * step + 13]; // R (swapped)
	float x13 = A[tid * step + 12] - A[tid * step + 4]; // I (swapped)

	float x6 = A[tid * step + 6] + A[tid * step + 14]; // R
	float x7 = A[tid * step + 7] + A[tid * step + 15]; // I
	float x14 = A[tid * step + 6] - A[tid * step + 14]; // R
	float x15 = A[tid * step + 7] - A[tid * step + 15]; // I

	// rotations
	x10 = x10 * coef;
	x11 = x11 * coef;

	float temp = x10 + x11;
	x11 = x11 - x10;
	x10 = temp;


	x14 = x14 * coef;
	x15 = x15 * coef;

	temp = -x14 + x15;
	x15 = -x15 - x14;
	x14 = temp;

	// stage 2
	// butterflies
	float tempR;
	float tempI;
	tempR = x0 + x4; // R
	tempI = x1 + x5; // I
	x4 = x0 - x4; // R
	x5 = x1 - x5; // I
	x0 = tempR;
	x1 = tempI;

	tempR = x2 + x6; // R
	tempI = x3 + x7; // I
	float tempR2 = x3 - x7;
	x7 = x6 - x2; // I (swapped)
	x6 = tempR2; // R (swapped)
	x2 = tempR;
	x3 = tempI;

	tempR = x8 + x12; // R
	tempI = x9 + x13; // I
	x12 = x8 - x12; // R
	x13 = x9 - x13; // I
	x8 = tempR;
	x9 = tempI;

	tempR = x10 + x14; // R
	tempI = x11 + x15; // I
	tempR2 = x11 - x15;
	x15 = x14 - x10; // I (swapped)
	x14 = tempR2; // R (swapped)
	x10 = tempR;
	x11 = tempI;

	// stage 3
	A[tid * step + 0] = x0 + x2;
	A[tid * step + 1] = x1 + x3;
	A[tid * step + 2] = x8 + x10;
	A[tid * step + 3] = x9 + x11;

	A[tid * step + 4] = x4 + x6;
	A[tid * step + 5] = x5 + x7;
	A[tid * step + 6] = x12 + x14;
	A[tid * step + 7] = x13 + x15;

	A[tid * step + 8] = x0 - x2;
	A[tid * step + 9] = x1 - x3;
	A[tid * step + 10] = x8 - x10;
	A[tid * step + 11] = x9 - x11;

	A[tid * step + 12] = x4 - x6;
	A[tid * step + 13] = x5 - x7;
	A[tid * step + 14] = x12 - x14;
	A[tid * step + 15] = x13 - x15;
}

__global__  void fft(float* A)
{
	execute_8point_fft(A);
}

int  main()
{
#define N 8

	float A[2 * N];
	float* Ad;

	int memsize = 2 * N * sizeof(float);


	for (int i = 0; i < N; i++)
	{
		A[2 * i] = i;
		A[2 * i + 1] = i;
	}

	cudaMalloc((void**)&Ad, memsize);

	cudaMemcpy(Ad, A, memsize, cudaMemcpyHostToDevice);

	dim3 gridDim(1, 1);
	dim3 blockDim(1, 1);
	fft <<<gridDim, blockDim>>> (Ad);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);


	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++) {
		printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);
	}

}
