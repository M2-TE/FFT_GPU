#pragma once

#include <vector>
#include <random>
#include <sstream>
#include <assert.h>
#include <iostream>
#include <math_constants.h>

enum class InitType { eGradient, eRandom, eWave };
template<typename T = float> struct FFTComplex
{
	T real;
	T imag;
};
template<typename T = float> class FFTData
{
public:
	FFTData() = default;
	FFTData(size_t nComplex, InitType type)
	{
		init(nComplex, type);
	}
	FFTData(const FFTData<double>& other)
	{
		vals.resize(other.vals.size());
		for (size_t i = 0; i < other.vals.size(); i++) {
			vals[i].real = (T)other.vals[i].real;
			vals[i].imag = (T)other.vals[i].imag;
		}
	}
	FFTData(const FFTData<float>& other)
	{
		vals.resize(other.vals.size());
		for (size_t i = 0; i < other.vals.size(); i++) {
			vals[i].real = (T)other.vals[i].real;
			vals[i].imag = (T)other.vals[i].imag;
		}
	}
	~FFTData()
	{
		if (bAllocated) deallocate();
	}

public:
	// init using given algorithm
	FFTData& init(size_t nComplex, InitType type)
	{
		switch (type) {
			case InitType::eGradient: init_a(nComplex); break;
			case InitType::eRandom: init_b(nComplex); break;
			case InitType::eWave: init_c(nComplex); break;
		}

		return *this;
	}
	// fill with values (real == imag) ranging from 0 to nComplex (exclusive) in order
	FFTData& init_a(size_t nComplex)
	{
		if (bAllocated) deallocate();

		vals.resize(nComplex);
		for (size_t i = 0; i < nComplex; i++)
		{
			T value = static_cast<T>(i);
			vals[i] = { value, value };
		}

		return *this;
	}
	// random values between min and max
	FFTData& init_b(size_t nComplex, float min = -1.0f, float max = 1.0f)
	{
		if (bAllocated) deallocate();

		std::random_device dev;
		std::mt19937 rng(dev());
		std::uniform_real_distribution<T> dist(min, max);

		vals.resize(nComplex);
		for (size_t i = 0; i < nComplex; i++)
		{
			vals[i] = { dist(rng), dist(rng) };
		}

		return *this;
	}
	// sin/cos
	FFTData& init_c(size_t nComplex)
	{
		if (bAllocated) deallocate();

		vals.resize(nComplex);
		for (size_t i = 0; i < nComplex; i++)
		{
			T real = cos((((T)2) * ((T)CUDART_PI_F) * ((T)i)) / ((T)nComplex));
			T imag = sin((((T)2) * ((T)CUDART_PI_F) * ((T)i)) / ((T)nComplex));
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
		auto error = cudaMemcpy(deviceData, vals.data(), sizeof(FFTComplex<T>) * vals.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
		return *this;
	}
	// download data from device
	FFTData& download()
	{
		auto error = cudaMemcpy(vals.data(), deviceData, sizeof(FFTComplex<T>) * vals.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
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

private:
	inline void allocate()
	{
		cudaMalloc((void**)&deviceData, sizeof(FFTComplex<T>) * vals.size());
	}
	inline void deallocate()
	{
		cudaFree((void**)&deviceData);
	}

public:
	std::vector<FFTComplex<T>> vals;
	T* deviceData = nullptr;

private:
	bool bAllocated = false;
};

// compares host-side data per-component
static void compare_fft(FFTData<float>& a, FFTData<float>& b)
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

	float diffAvg = diffTotal / (float)(a.vals.size() * 2);
	printf("Max difference: %.6f\n", diffMax);
	printf("Avg difference: %.6f\n", diffAvg);
}
// compares host-side data per-component
static void compare_fft(FFTData<float>& a, FFTData<double>& b)
{
	assert(a.vals.size() == b.vals.size(), "sizes do not match");

	double diffTotal = 0.0f;
	double diffMax = 0.0f;

	float* pA = reinterpret_cast<float*>(a.vals.data());
	double* pB = reinterpret_cast<double*>(b.vals.data());

	for (size_t i = 0; i < a.vals.size() * 2; i++, pA++, pB++) {
		double diff = 
			((double)*pA > *pB) ?
			((double)*pA - *pB) : 
			((double)*pA - *pB);
		diff = abs(diff);

		diffTotal += diff;
		diffMax = diffMax < diff ? diff : diffMax;
	}

	double diffAvg = diffTotal / (double)(a.vals.size() * 2);
	printf("Max difference: %.6f\n", diffMax);
	printf("Avg difference: %.6f\n", diffAvg);
}
