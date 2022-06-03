#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <math_constants.h>

typedef unsigned int uint;
#define INPUT_SIZE 128
#define WORD_SIZE 8
#define TEMPLATE template <uint wordSize = WORD_SIZE, uint fftSize = 64>
#define INDEXING_ALIASES const uint idx = threadIdx.x; const uint idy = threadIdx.y
#define STEPPING_ALIASES const uint xStep = wordSize * 2; const uint yStep = fftSize * 2

__device__ void mem_transfer(float* src, float* dst)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// TODO: make this read data in coalescence (irrelevant with only one active thread)
	// note: "perfect" coalescence would require [threads = wordSize * 2 * threadsPerBlock]
	uint index = idy * yStep + idx * xStep;
	for (uint i = 0; i < 16; i++) {
		dst[index + i] = src[index + i];
	}
}
__device__ void debug_values(float* S)
{
	const uint wordSize = 8;
	const uint step = wordSize * 2;
	const uint tid = threadIdx.x;

	for (uint i = 0; i < 8; i++) {
		printf("[Thread %d Value %d]\tReal: %f\t\tImag: %f\n", tid, i, S[tid * step + i * 2], S[tid * step + i * 2 + 1]);
	}
}

/// deprecated atm
__device__ void execute_8point_fft_deprecated(float* IN)
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
__device__ void execute_4point_fft_deprecated(float* IN)
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
__device__ void execute_2point_fft_deprecated(float* IN)
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
///

TEMPLATE __device__ void execute_8point_fft(float* S)
{
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	uint index = idx * xStep + idy * yStep;
	const float coef = sqrtf(2.0f) / 2.0f;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	// stage 1
	{
		// butterflies
		x0  = S[index +  0] + S[index +  8]; // R
		x1  = S[index +  1] + S[index +  9]; // I
		x8  = S[index +  0] - S[index +  8]; // R
		x9  = S[index +  1] - S[index +  9]; // I

		x2  = S[index +  2] + S[index + 10]; // R
		x3  = S[index +  3] + S[index + 11]; // I
		x10 = S[index +  2] - S[index + 10]; // R
		x11 = S[index +  3] - S[index + 11]; // I

		x4  = S[index +  4] + S[index + 12]; // R
		x5  = S[index +  5] + S[index + 13]; // I
		x12 = S[index +  5] - S[index + 13]; // R (swapped)
		x13 = S[index + 12] - S[index +  4]; // I (swapped)

		x6  = S[index +  6] + S[index + 14]; // R
		x7  = S[index +  7] + S[index + 15]; // I
		x14 = S[index +  6] - S[index + 14]; // R
		x15 = S[index +  7] - S[index + 15]; // I

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
		// butterflies (with bit reversal)
		S[index +  0] =  x0 +  x2;
		S[index +  1] =  x1 +  x3;
		S[index +  2] =  x8 + x10;
		S[index +  3] =  x9 + x11;

		S[index +  4] =  x4 +  x6;
		S[index +  5] =  x5 +  x7;
		S[index +  6] = x12 + x14;
		S[index +  7] = x13 + x15;

		S[index +  8] =  x0 -  x2;
		S[index +  9] =  x1 -  x3;
		S[index + 10] =  x8 - x10;
		S[index + 11] =  x9 - x11;

		S[index + 12] =  x4 -  x6;
		S[index + 13] =  x5 -  x7;
		S[index + 14] = x12 - x14;
		S[index + 15] = x13 - x15;
	}
}
TEMPLATE __device__ void shuffle(float* S)
{
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[wordSize * 2];
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// shuffle index bits (b6, b5, b4) <-> (b3, b2, b1) + (b0)
		uint index = idx * xStep + i;
		uint upper = index & 0b111'000'0;
		uint lower = index & 0b000'111'0;
		index = (upper >> 3) | (lower << 3);
		index += idy * yStep;

		// write both real and imag parts to temp
		temps[i]     = S[index];
		temps[i + 1] = S[index + 1];
	}

	// then write values using temp array
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = idx * xStep + idy * yStep + i;
		S[index]     = temps[i];
		S[index + 1] = temps[i + 1];
	}
}
TEMPLATE __device__ void rotate(float* S)
{
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2 * CUDART_PI_F / fftSize;

	for (uint i = 0; i < wordSize; i++) {

		float a = (float)idx;	// floor(index / 8) tid is basically that, no need for more calculations
		float b = (float)i;		// mod(i, 8) i is already guarenteed to be between 0 and 8
		float phi = a * b;
		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = idx * xStep + idy * yStep + i * 2;
		float real = S[index];
		float imag = S[index + 1];
		S[index]	 = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}

__global__ void fft(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle
	shuffle(S);

	// executing first 8-point FFT
	execute_8point_fft(S);

	// rotation + shuffle
	rotate(S);
	shuffle(S);

	// executing second 8-point FFT
	execute_8point_fft(S);

	// output shuffle
	shuffle(S);

	// transfer from shared to global memory
	mem_transfer(S, OUT);
}

int main()
{
	float* pIN;
	float IN[2 * INPUT_SIZE];
	float OUT[2 * INPUT_SIZE];

	for (int i = 0; i < INPUT_SIZE; i++)
	{
		IN[2 * i] = i;
		IN[2 * i + 1] = i;
	}

	// DEBUGGING for advanced indexing
	int j = 0;
	for (int i = 64; i < INPUT_SIZE; i++, j++)
	{
		IN[2 * i] = j;
		IN[2 * i + 1] = j;
	}

	int memsize = 2 * INPUT_SIZE * sizeof(float);
	cudaMalloc((void**)&pIN, memsize);
	cudaMemcpy(pIN, IN, memsize, cudaMemcpyHostToDevice);

	// Gets rid of false flags with IntelliSense
#ifdef __CUDACC__
	#define KERNEL_GRID(grid, block) <<< grid, block >>>
#else
	#define KERNEL_GRID(grid, block)
#endif
	dim3 gridDim(1, 1, 1);
	dim3 blockDim(8, INPUT_SIZE / 64, 1);

	fft KERNEL_GRID(gridDim, blockDim)(pIN, pIN);
	cudaMemcpy(OUT, pIN, memsize, cudaMemcpyDeviceToHost);


	printf("The  outputs are: \n");
	for (int l = 0; l < INPUT_SIZE; l++) {
		printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, OUT[2 * l], 2 * l + 1, OUT[2 * l + 1]);
	}

}
