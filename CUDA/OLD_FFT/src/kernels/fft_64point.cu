#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <math_constants.h>

// Gets rid of false flags with IntelliSense
#ifdef __CUDACC__
	#define KERNEL_GRID(grid, block) <<< grid, block >>>
#else
	#define KERNEL_GRID(grid, block)
#endif

//#define SINGLE_DEBUG


typedef unsigned int uint;
#define INPUT_SIZE 4096 // number of complex values
#define INPUTS_PER_CHUNK 64
#define THREADS_PER_CHUNK 8

// TODO: constexpr
#define INDEXING_ALIASES const uint idx = threadIdx.x; const uint idy = threadIdx.y
#define STEPPING_ALIASES const uint xStep = wordSize * 2; const uint yStep = fftSize * 2
#define TEMPLATE_A template <uint scal>
#define TEMPLATE_B template <uint nBits>
#define FFT_SHUFFLING template <uint inputShuffleSize = 0, uint outputShuffleSize = 0>

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

// bit reversal
TEMPLATE_B __device__ uint reverse_bits(uint val)
{
	// NOTE: standard bit reversal techniques either dont work on gpu
	// or are just really unperformant, so i used the CUDA intrinsic __brev() instead
	// however, it always operates on the full 32 bits of a value, so it needs to be manually adjusted
	// to work with only x bits (x = 7 bits in the 64-fft case)

	// shift bits to the major part
	// to only reverse a selected range of bits
	//return __brev(val << (sizeof(uint) * 8 - nBits));
	return __brev(val << (32 - nBits));
}
TEMPLATE_B __device__ void reverse_index_bits(float* S)
{
	constexpr uint wordSize = 8;
	constexpr uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// mask everything outside the relevant (e.g. 0-127) bits
	uint invertedMask = (0xff'ff'ff'ff >> (32 - (nBits + 1))) ^ 0xff'ff'ff'ff;

	// need to store values in temp array before writing
	float temps[wordSize * 2];
	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// TODO: work in 0-63 space instead of 0-127?

		// obtain relevant bits via mask
		//printf("Before: %d\n", (offset + i));
		uint index = offset + i;
		uint indexB = index & invertedMask;
		//printf("Invert: %d\n", indexB);
		index = reverse_bits<nBits>(index >> 1); // shift right by one to ignore R/I bit
		index = (index << 1) + indexB; // shift back to the left and add leftover

		//printf("After: %d\n", index);



		// write both real and imag parts to temp
		temps[i] = S[index];
		temps[i + 1] = S[index + 1];
	}

	__syncthreads();

	// then write values using temp array
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = offset + i;
		S[index] = temps[i];
		S[index + 1] = temps[i + 1];
	}

}

// shuffling
__device__ void shuffle_forward(float* S)
{
	const uint wordSize = 8;
	const uint fftSize = 64;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[wordSize * 2];
	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < wordSize * 2; i += 2) {

		// shuffle index bits
		uint index = offset + i;
		uint upper = index & 0b0'111111'000000'0;
		uint lower = index & 0b0'000000'111111'0;
		index = (upper >> 6) | (lower << 6);

		// write both real and imag parts to temp
		temps[i]     = S[index];
		temps[i + 1] = S[index + 1];
	}
	__syncthreads();

	// then write values using temp array
	for (uint i = 0; i < wordSize * 2; i += 2) {
		uint index = offset + i;
		S[index]     = temps[i];
		S[index + 1] = temps[i + 1];
	}
}

// rotations/ffts
TEMPLATE_A __device__ void rotate(float* S)
{
	const uint fftSize = 64;
	const uint wordSize = 8;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)(scal);

	for (uint i = 0; i < wordSize; i++) {

		// can put in constant memory -> cache
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
FFT_SHUFFLING __device__ void fft_2_point(float* S)
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
		S[outputShuffleSize * 4 + offsetR] = x2;
		S[outputShuffleSize * 4 + offsetI] = x3;
	}
	else {
		S[idx * xStep + 0] = x0;
		S[idx * xStep + 1] = x1;
		S[idx * xStep + 2] = x2;
		S[idx * xStep + 3] = x3;
	}
}
FFT_SHUFFLING __device__ void fft_4_point(float* S)
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
FFT_SHUFFLING __device__ void fft_8_point(float* S)
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
__device__ void fft_16_point(float* S)
{
	// input shuffle + first 8-point fft
	fft_8_point<2, false>(S);

	// single rotation for each value
	rotate<16>(S);

	// input shuffle + second fft (4x 2-point) + output shuffle
	fft_2_point<8, 4>(S + 0);
	fft_2_point<8, 4>(S + 4);
	fft_2_point<8, 4>(S + 8);
	fft_2_point<8, 4>(S + 12);
}
__device__ void fft_32_point(float* S)
{
	// input shuffle + first 8-point fft
	fft_8_point<4, false>(S);

	// single rotation for each value
	rotate<32>(S);

	// input shuffle + second fft (2x 4-point) + output shuffle
	fft_4_point<8, 4>(S + 0);
	fft_4_point<8, 4>(S + 8);
}
__device__ void fft_64_point(float* S)
{
	// input shuffle + first 8-point fft
	fft_8_point<8, false>(S);

	// single rotation for each value
	rotate<64>(S);

	// input shuffle + second 8-point fft + output shuffle
	fft_8_point<8, 8>(S);
}
__device__ void fft_4096_point(float* S)
{
	shuffle_forward(S);

	if constexpr (true) {
		__syncthreads();
		fft_64_point(S);
	}

	if constexpr (true) {
		__syncthreads();
		rotate<4096>(S);
	}

	__syncthreads();
	shuffle_forward(S);

	if constexpr (false) {
		__syncthreads();
		fft_64_point(S);
	}

	__syncthreads();
	//shuffle_forward(S);
}

// core kernel
__global__ void fft(float* IN, float* OUT)
{
	__shared__ float S[INPUT_SIZE * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);

	__syncthreads();
	fft_4096_point(S);
	__syncthreads();

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
		//IN[2 * i + 0] = i % 64; IN[2 * i + 1] = i % 64;
		IN[2 * i + 0] = i; IN[2 * i + 1] = i;
	}

	int memsize = 2 * INPUT_SIZE * sizeof(float);
	cudaMalloc((void**)&pIN, memsize);
	cudaMemcpy(pIN, IN, memsize, cudaMemcpyHostToDevice);

	dim3 gridDim(1, 1, 1);

#ifdef SINGLE_DEBUG
	dim3 blockDim(1, 1, 1);
#else
	dim3 blockDim(THREADS_PER_CHUNK, INPUT_SIZE / INPUTS_PER_CHUNK, 1);
#endif
	printf("Launching kernel with %d threads per chunk, %d chunks\n", blockDim.x, blockDim.y);
	fft KERNEL_GRID(gridDim, blockDim)(pIN, pIN);

	cudaMemcpy(OUT, pIN, memsize, cudaMemcpyDeviceToHost);
	printf("The  outputs are: \n");
	for (int l = 0; l < INPUT_SIZE; l++) {

#ifndef SINGLE_DEBUG
		printf("RE:A[%d]=%10.2f\t\t\t, IM: A[%d]=%10.2f\t\t\t \n ", 2 * l, OUT[2 * l], 2 * l + 1, OUT[2 * l + 1]);
#endif
	}
}
