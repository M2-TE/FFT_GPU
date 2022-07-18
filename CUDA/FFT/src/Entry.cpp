// constexpr math library
//#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "fft_data.cuh"
#include "cufft_impl.cuh"
#include "custom_fft_impl.cuh"

int main()
{
	static constexpr uint N = 64; // N-point fft
	FFTData<float> data, cufftData;
	FFTData<double> cufftDataDouble;
	cufftDataDouble.init(N, InitType::eGradient);
	data = cufftData = cufftDataDouble;

	// custom fft
	{
		data.upload();
		perform_custom_fft<N>(data);
		data.download();
	}

	// cuFFT
	{
		cufftData.upload();
		perform_cufft(cufftData);
		cufftData.download();
	}

	// cuFFT double
	{
		cufftDataDouble.upload();
		perform_cufft_double(cufftDataDouble);
		cufftDataDouble.download();
	}

	//data.print();
	//cufftData.print();

	printf("\ncustom vs cuFFT:\n");
	compare_fft(data, cufftData);

	printf("\ncustom vs cuFFT double precision:\n");
	compare_fft(data, cufftDataDouble);

	printf("\ncuFFT vs cuFFT double precision:\n");
	compare_fft(cufftData, cufftDataDouble);
}