#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ void execute_8point_fft_shared(float* IN, float* OUT)
{
	__shared__ float S[16];

	const unsigned int tid = threadIdx.x;
	const float coef = sqrtf(2.0f) / 2.0f;
	const unsigned int wordSize = 8;
	const unsigned int step = wordSize * 2;

	// TODO: make this read data in coalescence (irrelevant with only one active thread)
	// note: "perfect" coalescence would require [threads = wordSize * 2 * threadsPerBlock]
	S[tid * step +  0] = IN[tid * step +  0];
	S[tid * step +  1] = IN[tid * step +  1];
	S[tid * step +  2] = IN[tid * step +  2];
	S[tid * step +  3] = IN[tid * step +  3];
	S[tid * step +  4] = IN[tid * step +  4];
	S[tid * step +  5] = IN[tid * step +  5];
	S[tid * step +  6] = IN[tid * step +  6];
	S[tid * step +  7] = IN[tid * step +  7];
	S[tid * step +  8] = IN[tid * step +  8];
	S[tid * step +  9] = IN[tid * step +  9];
	S[tid * step + 10] = IN[tid * step + 10];
	S[tid * step + 11] = IN[tid * step + 11];
	S[tid * step + 12] = IN[tid * step + 12];
	S[tid * step + 13] = IN[tid * step + 13];
	S[tid * step + 14] = IN[tid * step + 14];
	S[tid * step + 15] = IN[tid * step + 15];
	S[tid * step + 16] = IN[tid * step + 16];

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
	// stage 1
	{
		// butterflies
		x0 = S[0] + S[8]; // R
		x1 = S[1] + S[9]; // I
		x8 = S[0] - S[8]; // R
		x9 = S[1] - S[9]; // I

		x2 = S[2] + S[10]; // R
		x3 = S[3] + S[11]; // I
		x10 = S[2] - S[10]; // R
		x11 = S[3] - S[11]; // I

		x4 = S[4] + S[12]; // R
		x5 = S[5] + S[13]; // I
		x12 = S[5] - S[13]; // R (swapped)
		x13 = S[12] - S[4]; // I (swapped)

		x6 = S[6] + S[14]; // R
		x7 = S[7] + S[15]; // I
		x14 = S[6] - S[14]; // R
		x15 = S[7] - S[15]; // I

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
	}

	// stage 2
	{
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
	}

	// stage 3
	{
		// butterflies
		S[tid * step + 0] = x0 + x2;
		S[tid * step + 1] = x1 + x3;
		S[tid * step + 2] = x8 + x10;
		S[tid * step + 3] = x9 + x11;

		S[tid * step + 4] = x4 + x6;
		S[tid * step + 5] = x5 + x7;
		S[tid * step + 6] = x12 + x14;
		S[tid * step + 7] = x13 + x15;

		S[tid * step + 8] = x0 - x2;
		S[tid * step + 9] = x1 - x3;
		S[tid * step + 10] = x8 - x10;
		S[tid * step + 11] = x9 - x11;

		S[tid * step + 12] = x4 - x6;
		S[tid * step + 13] = x5 - x7;
		S[tid * step + 14] = x12 - x14;
		S[tid * step + 15] = x13 - x15;
	}

	// TODO
	// Memory Export to global memory (TODO: same as input, need coalescence here)
	OUT[tid * step +  0] = S[tid * step +  0];
	OUT[tid * step +  1] = S[tid * step +  1];
	OUT[tid * step +  2] = S[tid * step +  2];
	OUT[tid * step +  3] = S[tid * step +  3];
	OUT[tid * step +  4] = S[tid * step +  4];
	OUT[tid * step +  5] = S[tid * step +  5];
	OUT[tid * step +  6] = S[tid * step +  6];
	OUT[tid * step +  7] = S[tid * step +  7];
	OUT[tid * step +  8] = S[tid * step +  8];
	OUT[tid * step +  9] = S[tid * step +  9];
	OUT[tid * step + 10] = S[tid * step + 10];
	OUT[tid * step + 11] = S[tid * step + 11];
	OUT[tid * step + 12] = S[tid * step + 12];
	OUT[tid * step + 13] = S[tid * step + 13];
	OUT[tid * step + 14] = S[tid * step + 14];
	OUT[tid * step + 15] = S[tid * step + 15];
	OUT[tid * step + 16] = S[tid * step + 16];
}
__device__ void execute_8point_fft(float* IN)
{
	const unsigned int tid = threadIdx.x;
	const float coef = sqrtf(2.0f) / 2.0f;
	const unsigned int wordSize = 8;
	const unsigned int step = wordSize * 2;

	// stage 1
	// butterflies
	float x0 = IN[tid * step + 0] + IN[tid * step + 8]; // R
	float x1 = IN[tid * step + 1] + IN[tid * step + 9]; // I
	float x8 = IN[tid * step + 0] - IN[tid * step + 8]; // R
	float x9 = IN[tid * step + 1] - IN[tid * step + 9]; // I

	float x2 = IN[tid * step + 2] + IN[tid * step + 10]; // R
	float x3 = IN[tid * step + 3] + IN[tid * step + 11]; // I
	float x10 = IN[tid * step + 2] - IN[tid * step + 10]; // R
	float x11 = IN[tid * step + 3] - IN[tid * step + 11]; // I

	float x4 = IN[tid * step + 4] + IN[tid * step + 12]; // R
	float x5 = IN[tid * step + 5] + IN[tid * step + 13]; // I
	float x12 = IN[tid * step + 5] - IN[tid * step + 13]; // R (swapped)
	float x13 = IN[tid * step + 12] - IN[tid * step + 4]; // I (swapped)

	float x6 = IN[tid * step + 6] + IN[tid * step + 14]; // R
	float x7 = IN[tid * step + 7] + IN[tid * step + 15]; // I
	float x14 = IN[tid * step + 6] - IN[tid * step + 14]; // R
	float x15 = IN[tid * step + 7] - IN[tid * step + 15]; // I

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
	// butterflies
	IN[tid * step + 0] = x0 + x2;
	IN[tid * step + 1] = x1 + x3;
	IN[tid * step + 2] = x8 + x10;
	IN[tid * step + 3] = x9 + x11;

	IN[tid * step + 4] = x4 + x6;
	IN[tid * step + 5] = x5 + x7;
	IN[tid * step + 6] = x12 + x14;
	IN[tid * step + 7] = x13 + x15;

	IN[tid * step + 8] = x0 - x2;
	IN[tid * step + 9] = x1 - x3;
	IN[tid * step + 10] = x8 - x10;
	IN[tid * step + 11] = x9 - x11;

	IN[tid * step + 12] = x4 - x6;
	IN[tid * step + 13] = x5 - x7;
	IN[tid * step + 14] = x12 - x14;
	IN[tid * step + 15] = x13 - x15;
}
__device__ void execute_4point_fft(float* IN)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int wordSize = 4;
	const unsigned int step = wordSize * 2;

	// stage 1
	// butterflies
	float x0 = IN[tid * step + 0] + IN[tid * step + 4]; // R
	float x1 = IN[tid * step + 1] + IN[tid * step + 5]; // I
	float x4 = IN[tid * step + 0] - IN[tid * step + 4]; // R
	float x5 = IN[tid * step + 1] - IN[tid * step + 5]; // I

	float x2 = IN[tid * step + 2] + IN[tid * step + 6]; // R
	float x3 = IN[tid * step + 3] + IN[tid * step + 7]; // I
	float x6 = IN[tid * step + 3] - IN[tid * step + 7]; // R (swapped)
	float x7 = IN[tid * step + 6] - IN[tid * step + 2]; // I (swapped)

	// stage 2
	// butterflies
	IN[tid * step + 0] = x0 + x2;
	IN[tid * step + 1] = x1 + x3;
	IN[tid * step + 2] = x4 + x6;
	IN[tid * step + 3] = x5 + x7;

	IN[tid * step + 4] = x0 - x2;
	IN[tid * step + 5] = x1 - x3;
	IN[tid * step + 6] = x4 - x6;
	IN[tid * step + 7] = x5 - x7;
}
__device__ void execute_2point_fft(float* IN)
{
	//__shared__ float S[4];
	const unsigned int tid = threadIdx.x;
	const unsigned int wordSize = 2;
	const unsigned int step = wordSize * 2;

	// only really need to store 0 and 1 in registers
	// but compiler puts em all in registers anyways 
	// and this is just more readable:
	float x0 = IN[tid * step + 0];
	float x1 = IN[tid * step + 1];
	float x2 = IN[tid * step + 2];
	float x3 = IN[tid * step + 3];

	// stage 1
	// butterflies
	IN[tid * step + 0] = x0 + x2;
	IN[tid * step + 1] = x1 + x3;
	IN[tid * step + 2] = x0 - x2;
	IN[tid * step + 3] = x1 - x3;
}

__global__ void fft(float* IN, float* OUT)
{
	execute_8point_fft_shared(IN, OUT);
	//execute_8point_fft(IN);
	//execute_4point_fft(IN);
	//execute_2point_fft(IN);
}

#define INPUT_SIZE 16
#define WORD_SIZE 8
int main()
{
	static constexpr size_t N = INPUT_SIZE;

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


	// Gets rid of false flags with IntelliSense
#ifdef __CUDACC__
	#define KERNEL_GRID(grid, block) <<< grid, block >>>
#else
	#define KERNEL_GRID(grid, block)
#endif
	dim3 gridDim(1, 1, 1);
	dim3 blockDim(INPUT_SIZE / WORD_SIZE, 1, 1);
	fft KERNEL_GRID(gridDim, blockDim)(Ad, Ad);
	cudaMemcpy(A, Ad, memsize, cudaMemcpyDeviceToHost);


	printf("The  outputs are: \n");
	for (int l = 0; l < N; l++) {
		printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, A[2 * l], 2 * l + 1, A[2 * l + 1]);
	}

}
