#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <math_constants.h>

typedef unsigned int uint;
#define THREADS_PER_CHUNK 8 // should be 8
#define CHUNK_SIZE 64 // should be 64
#define INPUT_SIZE 64 // number of complex values

#define INDEXING_ALIASES const uint idx = threadIdx.x; const uint idy = threadIdx.y
#define STEPPING_ALIASES const uint xStep = wordSize * 2; const uint yStep = fftSize * 2
#define TEMPLATE_A template <uint fftSize>
#define TEMPLATE_B template <uint inputShuffleSize = 0, uint outputShuffleSize = 0>

// utils
__device__ void debug_values(float* S)
{
	const uint wordSize = 4;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	for (uint i = 0; i < 8; i++) {
		uint index = idx * xStep + i * 2;
		printf("[Thread %d Value %d]\tReal: %f\t\tImag: %f\n", idx, i, S[index], S[index + 1]);
	}
}
__device__ void mem_transfer(float* src, float* dst)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// TODO: make this read/write data in coalescence
	uint index = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i++) {
		dst[index + i] = src[index + i];
	}
}
// shuffling variants (deprecated)
__device__ void shuffle(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[wordSize * 2];
	uint offsetSrc = idx * xStep; // could abstain from storing this in register, its only usage will be in offsetSrc + i, which is a single multiply+add operation
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// shuffle index bits (b6, b5, b4) <-> (b3, b2, b1) + (b0)
		uint index = offsetSrc + i;
		uint upper = index & 0b111'000'0;
		uint lower = index & 0b000'111'0;
		index = (upper >> 3) | (lower << 3);
		index += idy * yStep;

		// write both real and imag parts to temp
		temps[i]     = S[index];
		temps[i + 1] = S[index + 1];
	}

	// then write values using temp array
	uint offsetDst = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = offsetDst + i;
		S[index]     = temps[i];
		S[index + 1] = temps[i + 1];
	}
}
__device__ void shuffleB(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	float temp[wordSize * 2];
	uint offsetSrc = idx * 2 + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// read value at target (shuffled) index
		uint iSrc = i * wordSize + offsetSrc;
		temp[i] = S[iSrc];
		temp[i + 1] = S[iSrc + 1];

		//S[iDst]	  = S[iSrc];
		//S[iDst + 1] = S[iSrc + 1];
	}
	uint offsetDst = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// write value to src (pre-shuffle) index
		uint iDst = i + offsetDst;
		S[iDst] = temp[i];
		S[iDst + 1] = temp[i + 1];
	}
}
// rotations and stuff
TEMPLATE_A __device__ void rotate(float* S)
{
	const uint wordSize = 8;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2 * CUDART_PI_F / fftSize;

	for (uint i = 0; i < wordSize; i++) {

		float a = (float)idx;	// floor(index / 8) tid is basically that, no need for more calculations
		float b = (float)i; 	// mod(i, 8) i is already guarenteed to be between 0 and 8
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
TEMPLATE_B __device__ void execute_8point_fft_shuffled(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	// stage 1
	{
		// butterflies
		if constexpr (inputShuffleSize) {
			const uint offsetR = idx * 2 + idy * yStep;
			const uint offsetI = offsetR + 1;
			x0 =  S[inputShuffleSize *  0 + offsetR] + S[inputShuffleSize *  8 + offsetR]; // R
			x1 =  S[inputShuffleSize *  0 + offsetI] + S[inputShuffleSize *  8 + offsetI]; // I
			x8 =  S[inputShuffleSize *  0 + offsetR] - S[inputShuffleSize *  8 + offsetR]; // R
			x9 =  S[inputShuffleSize *  0 + offsetI] - S[inputShuffleSize *  8 + offsetI]; // I

			x2 =  S[inputShuffleSize *  2 + offsetR] + S[inputShuffleSize * 10 + offsetR]; // R
			x3 =  S[inputShuffleSize *  2 + offsetI] + S[inputShuffleSize * 10 + offsetI]; // I
			x10 = S[inputShuffleSize *  2 + offsetR] - S[inputShuffleSize * 10 + offsetR]; // R
			x11 = S[inputShuffleSize *  2 + offsetI] - S[inputShuffleSize * 10 + offsetI]; // I

			x4 =  S[inputShuffleSize *  4 + offsetR] + S[inputShuffleSize * 12 + offsetR]; // R
			x5 =  S[inputShuffleSize *  4 + offsetI] + S[inputShuffleSize * 12 + offsetI]; // I
			x12 = S[inputShuffleSize *  4 + offsetI] - S[inputShuffleSize * 12 + offsetI]; // R (swapped)
			x13 = S[inputShuffleSize * 12 + offsetR] - S[inputShuffleSize *  4 + offsetR]; // I (swapped)

			x6 =  S[inputShuffleSize *  6 + offsetR] + S[inputShuffleSize * 14 + offsetR]; // R
			x7 =  S[inputShuffleSize *  6 + offsetI] + S[inputShuffleSize * 14 + offsetI]; // I
			x14 = S[inputShuffleSize *  6 + offsetR] - S[inputShuffleSize * 14 + offsetR]; // R
			x15 = S[inputShuffleSize *  6 + offsetI] - S[inputShuffleSize * 14 + offsetI]; // I
		}
		else {
			uint index = idx * xStep + idy * yStep;
			x0 =  S[index +  0] + S[index +  8]; // R
			x1 =  S[index +  1] + S[index +  9]; // I
			x8 =  S[index +  0] - S[index +  8]; // R
			x9 =  S[index +  1] - S[index +  9]; // I
				  			 
			x2 =  S[index +  2] + S[index + 10]; // R
			x3 =  S[index +  3] + S[index + 11]; // I
			x10 = S[index +  2] - S[index + 10]; // R
			x11 = S[index +  3] - S[index + 11]; // I
							 
			x4 =  S[index +  4] + S[index + 12]; // R
			x5 =  S[index +  5] + S[index + 13]; // I
			x12 = S[index +  5] - S[index + 13]; // R (swapped)
			x13 = S[index + 12] - S[index +  4]; // I (swapped)

			x6 =  S[index +  6] + S[index + 14]; // R
			x7 =  S[index +  7] + S[index + 15]; // I
			x14 = S[index +  6] - S[index + 14]; // R
			x15 = S[index +  7] - S[index + 15]; // I
		}

		// rotations
		{
			const float coef = sqrtf(2.0f) / 2.0f;

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
		if constexpr (outputShuffleSize) {
			const uint offsetR = idx * 2 + idy * yStep;
			const uint offsetI = offsetR + 1;
			
			S[outputShuffleSize *  0 + offsetR] =  x0 +  x2;
			S[outputShuffleSize *  0 + offsetI] =  x1 +  x3;
			S[outputShuffleSize *  2 + offsetR] =  x8 + x10;
			S[outputShuffleSize *  2 + offsetI] =  x9 + x11;

			S[outputShuffleSize *  4 + offsetR] =  x4 +  x6;
			S[outputShuffleSize *  4 + offsetI] =  x5 +  x7;
			S[outputShuffleSize *  6 + offsetR] = x12 + x14;
			S[outputShuffleSize *  6 + offsetI] = x13 + x15;

			S[outputShuffleSize *  8 + offsetR] =  x0 -  x2;
			S[outputShuffleSize *  8 + offsetI] =  x1 -  x3;
			S[outputShuffleSize * 10 + offsetR] =  x8 - x10;
			S[outputShuffleSize * 10 + offsetI] =  x9 - x11;

			S[outputShuffleSize * 12 + offsetR] =  x4 -  x6;
			S[outputShuffleSize * 12 + offsetI] =  x5 -  x7;
			S[outputShuffleSize * 14 + offsetR] = x12 - x14;
			S[outputShuffleSize * 14 + offsetI] = x13 - x15;
		}
		else {
			uint index = idx * xStep + idy * yStep;
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
}
TEMPLATE_B __device__ void execute_4point_fft_shuffled(float* S)
{
	const uint wordSize = 4;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7;

	// stage 1
	// butterflies + rotations
	if constexpr (inputShuffleSize) {
		
		const uint offsetR = idx * 2 + idy * yStep;
		const uint offsetI = offsetR + 1;
		x0 = S[inputShuffleSize * 0 + offsetR] + S[inputShuffleSize * 4 + offsetR]; // R
		x1 = S[inputShuffleSize * 0 + offsetI] + S[inputShuffleSize * 4 + offsetI]; // I
		x4 = S[inputShuffleSize * 0 + offsetR] - S[inputShuffleSize * 4 + offsetR]; // R
		x5 = S[inputShuffleSize * 0 + offsetI] - S[inputShuffleSize * 4 + offsetI]; // I

		x2 = S[inputShuffleSize * 2 + offsetR] + S[inputShuffleSize * 6 + offsetR]; // R
		x3 = S[inputShuffleSize * 2 + offsetI] + S[inputShuffleSize * 6 + offsetI]; // I
		x6 = S[inputShuffleSize * 2 + offsetI] - S[inputShuffleSize * 6 + offsetI]; // R (swapped)
		x7 = S[inputShuffleSize * 6 + offsetR] - S[inputShuffleSize * 2 + offsetR]; // I (swapped)
	}
	else {
		const uint index = idx * xStep + idy * yStep;
		x0 = S[index + 0] + S[index + 4]; // R
		x1 = S[index + 1] + S[index + 5]; // I
		x4 = S[index + 0] - S[index + 4]; // R
		x5 = S[index + 1] - S[index + 5]; // I

		x2 = S[index + 2] + S[index + 6]; // R
		x3 = S[index + 3] + S[index + 7]; // I
		x6 = S[index + 3] - S[index + 7]; // R (swapped)
		x7 = S[index + 6] - S[index + 2]; // I (swapped)
	}

	// stage 2
	// butterflies + bit reversal
	if constexpr (outputShuffleSize) {

		const uint offsetR = idx * 2 + idy * yStep;
		const uint offsetI = offsetR + 1;
		S[outputShuffleSize *  0 + offsetR] = x0 + x2;
		S[outputShuffleSize *  0 + offsetI] = x1 + x3;
		S[outputShuffleSize *  4 + offsetR] = x4 + x6;
		S[outputShuffleSize *  4 + offsetI] = x5 + x7;

		S[outputShuffleSize *  8 + offsetR] = x0 - x2;
		S[outputShuffleSize *  8 + offsetI] = x1 - x3;
		S[outputShuffleSize * 12 + offsetR] = x4 - x6;
		S[outputShuffleSize * 12 + offsetI] = x5 - x7;
	}
	else {
		const uint index = idx * xStep + idy * yStep;
		S[index + 0] = x0 + x2;
		S[index + 1] = x1 + x3;
		S[index + 2] = x4 + x6;
		S[index + 3] = x5 + x7;

		S[index + 4] = x0 - x2;
		S[index + 5] = x1 - x3;
		S[index + 6] = x4 - x6;
		S[index + 7] = x5 - x7;
	}
}
TEMPLATE_B __device__ void execute_2point_fft_shuffled(float* S)
{
	const uint wordSize = 2;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3;

	// stage 1
	// butterflies
	if constexpr (inputShuffleSize) {
		const uint offsetR = idx * 2 + idy * yStep;
		const uint offsetI = offsetR + 1;

		x0 = S[inputShuffleSize * 0 + offsetR] + S[inputShuffleSize * 2 + offsetR]; // R
		x1 = S[inputShuffleSize * 0 + offsetI] + S[inputShuffleSize * 2 + offsetI]; // I

		x2 = S[inputShuffleSize * 0 + offsetR] - S[inputShuffleSize * 2 + offsetR]; // R
		x3 = S[inputShuffleSize * 0 + offsetI] - S[inputShuffleSize * 2 + offsetI]; // I
	}
	else {
		const uint index = idx * xStep + idy * yStep;
		x0 = S[index + 0] + S[index + 2]; // R
		x1 = S[index + 1] + S[index + 3]; // I

		x2 = S[index + 0] - S[index + 2]; // R
		x3 = S[index + 1] - S[index + 3]; // I
	}

	// output only
	if constexpr (outputShuffleSize) {
		const uint offsetR = idx * 2 + idy * yStep;
		const uint offsetI = offsetR + 1;

		S[outputShuffleSize * 0 + offsetR] = x0;
		S[outputShuffleSize * 0 + offsetI] = x1;

		S[outputShuffleSize * 2 + offsetR] = x2;
		S[outputShuffleSize * 2 + offsetI] = x3;
	}
	else {
		S[idx * xStep + 0] = x0;
		S[idx * xStep + 1] = x1;
		S[idx * xStep + 2] = x2;
		S[idx * xStep + 3] = x3;
	}
}

__device__ void shuffle_A(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[wordSize * 2];
	uint offsetSrc = idx * xStep; // could abstain from storing this in register, its only usage will be in offsetSrc + i, which is a single multiply+add operation
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// shuffle index bits (b6, b5, b4) <-> (b3, b2, b1) + (b0)
		uint index = offsetSrc + i;
		uint upper = index & 0b1'000'0;
		uint lower = index & 0b0'111'0;
		index = (upper >> 3) | (lower << 1);
		index += idy * yStep;

		// write both real and imag parts to temp
		temps[i] = S[index];
		temps[i + 1] = S[index + 1];
	}

	// then write values using temp array
	uint offsetDst = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = offsetDst + i;
		S[index] = temps[i];
		S[index + 1] = temps[i + 1];
	}
}
__device__ void shuffle_B(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[wordSize * 2];
	uint offsetSrc = idx * xStep; // could abstain from storing this in register, its only usage will be in offsetSrc + i, which is a single multiply+add operation
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// shuffle index bits (b6, b5, b4) <-> (b3, b2, b1) + (b0)
		uint index = offsetSrc + i;
		uint upper = index & 0b111'0'0;
		uint lower = index & 0b000'1'0;
		index = (upper >> 1) | (lower << 3);
		index += idy * yStep;

		// write both real and imag parts to temp
		temps[i] = S[index];
		temps[i + 1] = S[index + 1];
	}

	// then write values using temp array
	uint offsetDst = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = offsetDst + i;
		S[index] = temps[i];
		S[index + 1] = temps[i + 1];
	}
}

// core kernel
__global__ void fft(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle + first 8-point fft
	execute_8point_fft_shuffled<8, false>(S);

	// single rotation for each value
	rotate<64>(S);

	// input shuffle + second 8-point fft + output shuffle
	execute_8point_fft_shuffled<8, 8>(S);

	// transfer from shared to global memory
	mem_transfer(S, OUT);
}
__global__ void fft_32(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle + first 8-point fft
	execute_8point_fft_shuffled<4, false>(S);
	
	// single rotation for each value
	rotate<32>(S);

	// input shuffle + second fft (2x 4-point) + output shuffle
	// NOTE: shuffle size of 4 is normal shuffle, 8 is inverse
	execute_4point_fft_shuffled<8, 4>(S + 0);
	execute_4point_fft_shuffled<8, 4>(S + 8);

	// transfer from shared to global memory
	mem_transfer(S, OUT);
}
__global__ void fft_16(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle + first 8-point fft
	execute_8point_fft_shuffled<2, false>(S);

	// single rotation for each value
	rotate<16>(S);

	// input shuffle + second fft (2x 4-point) + output shuffle
	execute_2point_fft_shuffled<8, 2>(S + 0);
	execute_2point_fft_shuffled<8, 2>(S + 8);
	execute_2point_fft_shuffled<8, 2>(S + 16);
	execute_2point_fft_shuffled<8, 2>(S + 24);

	// transfer from shared to global memory
	mem_transfer(S, OUT);
}
__global__ void fft_old(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	// input shuffle
	shuffleB(S);
	// executing first 8-point FFT
	execute_8point_fft_shuffled<false, false>(S);

	// rotation + shuffle
	rotate<64>(S);
	shuffleB(S);

	// executing second 8-point FFT
	execute_8point_fft_shuffled<false, false>(S);

	// output shuffle
	shuffleB(S);

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
		// DEBUGGING for advanced indexing
		IN[2 * i] = i % 64;
		IN[2 * i + 1] = i % 64;
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
	dim3 blockDim(THREADS_PER_CHUNK, INPUT_SIZE / CHUNK_SIZE, 1);
	fft KERNEL_GRID(gridDim, blockDim)(pIN, pIN);
	//fft_32 KERNEL_GRID(gridDim, blockDim)(pIN, pIN);
	//fft_16 KERNEL_GRID(gridDim, blockDim)(pIN, pIN);

	cudaMemcpy(OUT, pIN, memsize, cudaMemcpyDeviceToHost);
	printf("The  outputs are: \n");
	for (int l = 0; l < INPUT_SIZE; l++) {
		printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, OUT[2 * l], 2 * l + 1, OUT[2 * l + 1]);
	}
}
