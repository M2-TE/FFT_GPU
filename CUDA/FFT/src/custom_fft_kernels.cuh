#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <math_constants.h>

#define FFT_SHUFFLING template <uint inputShuffleSize = 0, uint outputShuffleSize = 0>
// TODO: constexpr?
#define CONSTANT_ALIASES const uint nWordsPerChunk = 64;
#define INDEXING_ALIASES const uint idx = threadIdx.x; const uint idy = threadIdx.y
#define STEPPING_ALIASES const uint xStep = nWords * 2; const uint yStep = nWordsPerChunk * 2

// utils
__device__ void debug_values(float* S)
{
	const uint nWords = 4;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	for (uint i = 0; i < 8; i++) {
		uint index = idx * xStep + i * 2;
		printf("[Thread %d Value %d]\tReal: %f\t\tImag: %f\n", idx, i, S[index], S[index + 1]);
	}
}
__device__ void mem_transfer(float* src, float* dst)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// TODO: make this read/write data in coalescence
	uint index = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords * 2; i++) {
		dst[index + i] = src[index + i];
	}
}

// bit reversal
template<uint nBits> __device__ uint reverse_bits(uint val)
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
template<uint nBits> __device__ void reverse_index_bits(float* S)
{
	constexpr uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// mask everything outside the relevant (e.g. 0-127) bits
	uint invertedMask = (0xff'ff'ff'ff >> (32 - (nBits + 1))) ^ 0xff'ff'ff'ff;

	// need to store values in temp array before writing
	float temps[nWords * 2];
	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords * 2; i += 2) {

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
	for (uint i = 0; i < nWords * 2; i += 2) {
		uint index = offset + i;
		S[index] = temps[i];
		S[index + 1] = temps[i + 1];
	}

}

// rotations
__device__ void rotate_16(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)16;

	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)(idx % 2);
		float b = (float)i;
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_32(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)32;

	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)(idx % 4);
		float b = (float)i;
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_64(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)64;

	uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idx;
		float b = (float)i;
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_128(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)128;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_256(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)256;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_512(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)512;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_1024(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)1024;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_2048(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)2048;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}
__device__ void rotate_4096(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;
	const float scaling = 2.0f * CUDART_PI_F / (float)4096;

	const uint offset = idx * xStep + idy * yStep;
	for (uint i = 0; i < nWords; i++) {

		// can put in constant memory -> cache
		float a = (float)idy;			// a = 0 -> 32
		float b = (float)i + idx * 8; 	// b = 0 -> 63
		float phi = a * b;

		float ang = scaling * phi;
		float c = cosf(ang);
		float s = sinf(ang);

		uint index = i * 2 + offset;
		float real = S[index];
		float imag = S[index + 1];
		S[index] = c * real + s * imag;
		S[index + 1] = c * imag - s * real;
	}
}

// shuffling
template<uint nBitsUpper, uint nBitsLower> __device__ void shuffle(float* S)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	const uint upperMask = ((1 << nBitsUpper) - 1) << (1 + nBitsLower);
	const uint lowerMask = ((1 << nBitsLower) - 1) << 1;
	const uint offset = idx * xStep + idy * yStep;

	// need to store values in temp array before writing
	// (not all threads write all their 8 values at once -> undefined behaviour otherwise)
	float temps[nWords * 2];
	for (uint i = 0; i < nWords * 2; i += 2) {
		uint index = offset + i;
		temps[i] = S[index];
		temps[i + 1] = S[index + 1];
	}

	__syncthreads();
	// then write to shared memory
	for (uint i = 0; i < nWords * 2; i += 2) {

		// shuffle index bits
		uint index = offset + i;
		uint upper = index & upperMask;
		uint lower = index & lowerMask;
		index = (upper >> nBitsLower) | (lower << nBitsUpper);

		// write both real and imag parts to temp
		S[index] = temps[i];
		S[index + 1] = temps[i + 1];
	}
}

// fft kernels
FFT_SHUFFLING __device__ void fft_2_point(float* S, uint shuffleOffset = 0u)
{
	const uint nWords = 2;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3;

	// stage 1
	// butterflies
	if constexpr (inputShuffleSize) {
		const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
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
		const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
		const uint offsetI = offsetR + 1;
		S[outputShuffleSize * 0 + offsetR] = x0;
		S[outputShuffleSize * 0 + offsetI] = x1;
		S[outputShuffleSize * 2 + offsetR] = x2;
		S[outputShuffleSize * 2 + offsetI] = x3;
	}
	else {
		const uint index = idx * xStep + idy * yStep;
		S[index + 0] = x0;
		S[index + 1] = x1;
		S[index + 2] = x2;
		S[index + 3] = x3;
	}
}
FFT_SHUFFLING __device__ void fft_4_point(float* S, uint shuffleOffset = 0u)
{
	const uint nWords = 4;
	//const uint nWordsPerChunk = 32;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7;

	// stage 1
	// butterflies + rotations
	if constexpr (inputShuffleSize) {

		const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
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

		const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
		const uint offsetI = offsetR + 1;
		S[outputShuffleSize * 0 + offsetR] = x0 + x2;
		S[outputShuffleSize * 0 + offsetI] = x1 + x3;
		S[outputShuffleSize * 2 + offsetR] = x4 + x6;
		S[outputShuffleSize * 2 + offsetI] = x5 + x7;

		S[outputShuffleSize * 4 + offsetR] = x0 - x2;
		S[outputShuffleSize * 4 + offsetI] = x1 - x3;
		S[outputShuffleSize * 6 + offsetR] = x4 - x6;
		S[outputShuffleSize * 6 + offsetI] = x5 - x7;
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
FFT_SHUFFLING __device__ void fft_8_point(float* S, uint shuffleOffset = 0u)
{
	const uint nWords = 8;
	CONSTANT_ALIASES;
	INDEXING_ALIASES;
	STEPPING_ALIASES;

	// registers for the main data inbetween stages
	float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

	// stage 1
	{
		// butterflies
		if constexpr (inputShuffleSize) {
			const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
			const uint offsetI = offsetR + 1;
			x0 = S[inputShuffleSize * 0 + offsetR] + S[inputShuffleSize * 8 + offsetR]; // R
			x1 = S[inputShuffleSize * 0 + offsetI] + S[inputShuffleSize * 8 + offsetI]; // I
			x8 = S[inputShuffleSize * 0 + offsetR] - S[inputShuffleSize * 8 + offsetR]; // R
			x9 = S[inputShuffleSize * 0 + offsetI] - S[inputShuffleSize * 8 + offsetI]; // I

			x2 = S[inputShuffleSize * 2 + offsetR] + S[inputShuffleSize * 10 + offsetR]; // R
			x3 = S[inputShuffleSize * 2 + offsetI] + S[inputShuffleSize * 10 + offsetI]; // I
			x10 = S[inputShuffleSize * 2 + offsetR] - S[inputShuffleSize * 10 + offsetR]; // R
			x11 = S[inputShuffleSize * 2 + offsetI] - S[inputShuffleSize * 10 + offsetI]; // I

			x4 = S[inputShuffleSize * 4 + offsetR] + S[inputShuffleSize * 12 + offsetR]; // R
			x5 = S[inputShuffleSize * 4 + offsetI] + S[inputShuffleSize * 12 + offsetI]; // I
			x12 = S[inputShuffleSize * 4 + offsetI] - S[inputShuffleSize * 12 + offsetI]; // R (swapped)
			x13 = S[inputShuffleSize * 12 + offsetR] - S[inputShuffleSize * 4 + offsetR]; // I (swapped)

			x6 = S[inputShuffleSize * 6 + offsetR] + S[inputShuffleSize * 14 + offsetR]; // R
			x7 = S[inputShuffleSize * 6 + offsetI] + S[inputShuffleSize * 14 + offsetI]; // I
			x14 = S[inputShuffleSize * 6 + offsetR] - S[inputShuffleSize * 14 + offsetR]; // R
			x15 = S[inputShuffleSize * 6 + offsetI] - S[inputShuffleSize * 14 + offsetI]; // I
		}
		else {
			const uint index = idx * xStep + idy * yStep;
			x0 = S[index + 0] + S[index + 8]; // R
			x1 = S[index + 1] + S[index + 9]; // I
			x8 = S[index + 0] - S[index + 8]; // R
			x9 = S[index + 1] - S[index + 9]; // I

			x2 = S[index + 2] + S[index + 10]; // R
			x3 = S[index + 3] + S[index + 11]; // I
			x10 = S[index + 2] - S[index + 10]; // R
			x11 = S[index + 3] - S[index + 11]; // I

			x4 = S[index + 4] + S[index + 12]; // R
			x5 = S[index + 5] + S[index + 13]; // I
			x12 = S[index + 5] - S[index + 13]; // R (swapped)
			x13 = S[index + 12] - S[index + 4]; // I (swapped)

			x6 = S[index + 6] + S[index + 14]; // R
			x7 = S[index + 7] + S[index + 15]; // I
			x14 = S[index + 6] - S[index + 14]; // R
			x15 = S[index + 7] - S[index + 15]; // I
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
			const uint offsetR = idx * 2 + idy * yStep + shuffleOffset;
			const uint offsetI = offsetR + 1;

			S[outputShuffleSize * 0 + offsetR] = x0 + x2;
			S[outputShuffleSize * 0 + offsetI] = x1 + x3;
			S[outputShuffleSize * 2 + offsetR] = x8 + x10;
			S[outputShuffleSize * 2 + offsetI] = x9 + x11;

			S[outputShuffleSize * 4 + offsetR] = x4 + x6;
			S[outputShuffleSize * 4 + offsetI] = x5 + x7;
			S[outputShuffleSize * 6 + offsetR] = x12 + x14;
			S[outputShuffleSize * 6 + offsetI] = x13 + x15;

			S[outputShuffleSize * 8 + offsetR] = x0 - x2;
			S[outputShuffleSize * 8 + offsetI] = x1 - x3;
			S[outputShuffleSize * 10 + offsetR] = x8 - x10;
			S[outputShuffleSize * 10 + offsetI] = x9 - x11;

			S[outputShuffleSize * 12 + offsetR] = x4 - x6;
			S[outputShuffleSize * 12 + offsetI] = x5 - x7;
			S[outputShuffleSize * 14 + offsetR] = x12 - x14;
			S[outputShuffleSize * 14 + offsetI] = x13 - x15;
		}
		else {
			const uint index = idx * xStep + idy * yStep;
			S[index + 0] = x0 + x2;
			S[index + 1] = x1 + x3;
			S[index + 2] = x8 + x10;
			S[index + 3] = x9 + x11;

			S[index + 4] = x4 + x6;
			S[index + 5] = x5 + x7;
			S[index + 6] = x12 + x14;
			S[index + 7] = x13 + x15;

			S[index + 8] = x0 - x2;
			S[index + 9] = x1 - x3;
			S[index + 10] = x8 - x10;
			S[index + 11] = x9 - x11;

			S[index + 12] = x4 - x6;
			S[index + 13] = x5 - x7;
			S[index + 14] = x12 - x14;
			S[index + 15] = x13 - x15;
		}
	}
}
__device__ void fft_16_point(float* S)
{
	uint offset = (threadIdx.x / 2) * (32 - 4);

	// input shuffle + first 8-point fft
	fft_8_point<2, false>(S, offset);

	// single rotation for each value
	rotate_16(S);

	// input shuffle + second fft (4x 2-point) + output shuffle
	fft_2_point<8, 8>(S, offset + 0);
	fft_2_point<8, 8>(S, offset + 4);
	fft_2_point<8, 8>(S, offset + 8);
	fft_2_point<8, 8>(S, offset + 12);
}
__device__ void fft_32_point(float* S)
{
	uint offset = (threadIdx.x / 4) * (64 - 8);

	// input shuffle + first 8-point fft
	fft_8_point<4, false>(S, offset);

	// single rotation for each value
	rotate_32(S);

	// input shuffle + second fft (2x 4-point) + output shuffle
	fft_4_point<8, 8>(S, offset);
	fft_4_point<8, 8>(S, offset + 8);
}
__device__ void fft_64_point(float* S)
{
	// input shuffle + first 8-point fft
	fft_8_point<8, false>(S);

	// single rotation for each value
	rotate_64(S);

	// input shuffle + second 8-point fft + output shuffle
	fft_8_point<8, 8>(S);
}
// TODO: manual shuffling on the 2/4/8 variants inside 128/256/512? prolly requires extra sync steps, but with added performance
__device__ void fft_128_point(float* S)
{
	shuffle<6, 1>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_128(S);

	__syncthreads();
	shuffle<1, 6>(S);

	__syncthreads();
	fft_2_point<false, false>(S);
	fft_2_point<false, false>(S + 32);
	fft_2_point<false, false>(S + 64);
	fft_2_point<false, false>(S + 96);

	__syncthreads();
	shuffle<6, 1>(S);
}
__device__ void fft_256_point(float* S)
{
	shuffle<6, 2>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_256(S);

	__syncthreads();
	shuffle<2, 6>(S);

	__syncthreads();
	fft_4_point<false, false>(S);
	fft_4_point<false, false>(S + 64);

	__syncthreads();
	shuffle<6, 2>(S);
}
__device__ void fft_512_point(float* S)
{
	shuffle<6, 3>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_512(S);

	__syncthreads();
	shuffle<3, 6>(S);

	__syncthreads();
	fft_8_point<false, false>(S);

	__syncthreads();
	shuffle<6, 3>(S);
}
//
__device__ void fft_1024_point(float* S)
{
	shuffle<6, 4>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_1024(S);

	__syncthreads();
	shuffle<4, 6>(S);

	__syncthreads();
	fft_16_point(S);

	__syncthreads();
	shuffle<6, 4>(S);
}
__device__ void fft_2048_point(float* S)
{
	shuffle<6, 5>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_2048(S);

	__syncthreads();
	shuffle<5, 6>(S);

	__syncthreads();
	fft_32_point(S);

	__syncthreads();
	shuffle<6, 5>(S);
}
__device__ void fft_4096_point(float* S)
{
	shuffle<6, 6>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	rotate_4096(S);

	__syncthreads();
	shuffle<6, 6>(S);

	__syncthreads();
	fft_64_point(S);

	__syncthreads();
	shuffle<6, 6>(S);
}

// core kernel
template <uint N> __global__ void fft(float* IN, float* OUT)
{
	__shared__ float S[N * 2];

	// transfer from global to shared memory
	mem_transfer(IN, S);
	if constexpr (N > 64) __syncthreads();

	// god this is a beautiful hell
	if		constexpr (N == 4096) fft_4096_point(S);
	else if constexpr (N == 2048) fft_2048_point(S);
	else if constexpr (N == 1024) fft_1024_point(S);
	else if constexpr (N ==  512) fft_512_point(S);
	else if constexpr (N ==  256) fft_256_point(S);
	else if constexpr (N ==  128) fft_128_point(S);
	else if constexpr (N ==   64) fft_64_point(S);
	else if constexpr (N ==   32) fft_32_point(S);
	else if constexpr (N ==   16) fft_16_point(S);
	else if constexpr (N ==    8) fft_8_point(S);
	else if constexpr (N ==    4) fft_4_point(S);
	else if constexpr (N ==    2) fft_2_point(S);
	else static_assert(false, "Invalid FFT size for static kernel");
	
	// transfer from shared to global memory
	if constexpr (N > 64) __syncthreads();
	mem_transfer(S, OUT);
}
