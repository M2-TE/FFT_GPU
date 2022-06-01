#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

typedef unsigned int uint;

template <uint nBits>
__device__ uint reverse_bits(uint val)
{
	// NOTE: standard bit reversal either doesnt work on gpu
	// or is just really unperformant, so i used the CUDA intrinsic __brev() instead
	// however, it always operates on the full 32 bits of a value, so it needs to be manually adjusted
	// to work with only x bits (x = 7 bits in the 64-fft case)
	
	// shift bits to the major part
	// to only reverse a selected range of bits
	return __brev(val << (sizeof(uint) * 8 - nBits));
}
__device__ void mem_transfer(float* src, float* dst)
{
	const uint wordSize = 8;
	const uint step = wordSize * 2;
	const uint tid = threadIdx.x;

	// TODO: make this read data in coalescence (irrelevant with only one active thread)
	// note: "perfect" coalescence would require [threads = wordSize * 2 * threadsPerBlock]
	for (uint i = 0; i < 16; i++) {
		dst[tid * step + i] = src[tid * step + i];
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

__device__ void execute_8point_fft(float* S)
{
	const uint tid = threadIdx.x;
	const uint wordSize = 8;
	const uint step = wordSize * 2;
	const float coef = sqrtf(2.0f) / 2.0f;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	// stage 1
	{
		// butterflies
		x0  = S[tid * step +  0] + S[tid * step +  8]; // R
		x1  = S[tid * step +  1] + S[tid * step +  9]; // I
		x8  = S[tid * step +  0] - S[tid * step +  8]; // R
		x9  = S[tid * step +  1] - S[tid * step +  9]; // I
		
		x2  = S[tid * step +  2] + S[tid * step + 10]; // R
		x3  = S[tid * step +  3] + S[tid * step + 11]; // I
		x10 = S[tid * step +  2] - S[tid * step + 10]; // R
		x11 = S[tid * step +  3] - S[tid * step + 11]; // I

		x4  = S[tid * step +  4] + S[tid * step + 12]; // R
		x5  = S[tid * step +  5] + S[tid * step + 13]; // I
		x12 = S[tid * step +  5] - S[tid * step + 13]; // R (swapped)
		x13 = S[tid * step + 12] - S[tid * step +  4]; // I (swapped)

		x6  = S[tid * step +  6] + S[tid * step + 14]; // R
		x7  = S[tid * step +  7] + S[tid * step + 15]; // I
		x14 = S[tid * step +  6] - S[tid * step + 14]; // R
		x15 = S[tid * step +  7] - S[tid * step + 15]; // I

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
}
__device__ void shuffle()
{

}
__device__ void rotate()
{

}
__global__ void fft(float* IN, float* OUT)
{
	__shared__ float S[64];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle
	shuffle();
	
	// executing first 8-point FFT
	execute_8point_fft(S);

	// rotation + shuffle or shuffle + rotation
	rotate();
	shuffle();

	// executing second 8-point FFT
	execute_8point_fft(S);

	// output shuffle
	shuffle();

	// transfer from shared to global memory
	mem_transfer(S, OUT);
}

#define INPUT_SIZE 64
#define WORD_SIZE 8
int main()
{
	//uint val = 0b0100'0001;
	//printf("%d\n", reverse_bits<7>(val));
	//return;

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
