#pragma once

#include <vector>
#include <random>
#include <sstream>
#include <assert.h>
#include <iostream>
#include <math_constants.h>

struct FFTComplex
{
	float real;
	float imag;
};

class FFTData
{
public:
	FFTData() = default;
	~FFTData()
	{
		if (bAllocated) deallocate();
	}

public:
	// fill with values (real == imag) ranging from 0 to nSize (exclusive) in order
	FFTData& init_a(size_t nSize)
	{
		if (bAllocated) deallocate();

		vals.resize(nSize);
		for (size_t i = 0; i < nSize; i++)
		{
			float value = static_cast<float>(i);
			//float value = static_cast<float>(i - 16);
			//if (i < 16) value = 0;
			vals[i] = { value, value };
		}

		return *this;
	}

	// random values between min and max
	FFTData& init_b(size_t nSize, float min, float max)
	{
		if (bAllocated) deallocate();

		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<float> dist(min, max);

		vals.resize(nSize);
		for (size_t i = 0; i < nSize; i++)
		{
			vals[i] = { dist(rng), dist(rng) };
		}

		return *this;
	}

	// sin/cos funcs
	FFTData& init_c(size_t nSize)
	{
		if (bAllocated) deallocate();

		vals.resize(nSize);
		for (size_t i = 0; i < nSize; i++)
		{
			float real = cosf((2.0f * asfloat(CUDART_PI_F) * asfloat(i)) / asfloat(nSize));
			float imag = sinf((2.0f * asfloat(CUDART_PI_F) * asfloat(i)) / asfloat(nSize));
			vals[i] = { real, imag };
		}
		return *this;
	}


public:
	inline size_t size() { return vals.size(); }
	// upload data to device
	FFTData& upload()
	{
		if (!bAllocated) allocate();
		auto error = cudaMemcpy(deviceData, vals.data(), sizeof(FFTComplex) * vals.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
		return *this;
	}
	// download data from device
	FFTData& download()
	{
		auto error = cudaMemcpy(vals.data(), deviceData, sizeof(FFTComplex) * vals.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
		return *this;
	}
	// print all complex values in sequence
	FFTData& print()
	{
		std::ostringstream oss;
		oss.precision(2);
		oss.setf(std::ios_base::fixed);

		for (size_t i = 0; i < vals.size(); i++) {
			oss << "[" << i << "]\tReal: " << vals[i].real << "\tImag: " << vals[i].imag << std::endl;
		}

		std::cout << oss.str();

		return *this;
	}
	// print complex value at given position
	FFTData& print(size_t i)
	{
		std::ostringstream oss;
		oss.precision(2);
		oss.setf(std::ios_base::fixed);

		oss << "[" << i << "]\tReal: " << vals[i].real << "\tImag: " << vals[i].imag << std::endl;

		std::cout << oss.str();

		return *this;
	}
	
	// compares host-side data per-component using given accuracy range
	static bool compare(FFTData& a, FFTData& b, float accuracy = 0.1f)
	{
		assert(a.vals.size() == b.vals.size(), "sizes do not match");

		float diffTotal = 0.0f;
		float diffMax = 0.0f;


		float* pA = reinterpret_cast<float*>(a.vals.data());
		float* pB = reinterpret_cast<float*>(b.vals.data());

		for (size_t i = 0; i < a.vals.size() * 2; i++, pA++, pB++) {
			float diff = (*pA > *pB) ? (*pA - *pB) : (*pA - *pB);
			diff = fabsf(diff);

			diffTotal += diff;
			diffMax = diffMax < diff ? diff : diffMax;
		}

		float diffAvg = diffTotal / asfloat(a.vals.size() * 2);
		printf("Max difference: %.6f\n", diffMax);
		printf("Avg difference: %.6f\n", diffAvg);

		return diffMax <= accuracy;
	}

private:
	inline void allocate()
	{
		cudaMalloc((void**)&deviceData, sizeof(FFTComplex) * vals.size());
	}
	inline void deallocate()
	{
		cudaFree((void**)&deviceData);
	}

public:
	std::vector<FFTComplex> vals;
	float* deviceData = nullptr;
private:
	bool bAllocated = false;
};