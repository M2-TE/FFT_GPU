// constexpr math library
//#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "fft_data.cuh"
#include "cufft_impl.cuh"
#include "custom_fft_impl.cuh"



int main()
{
	static constexpr uint N = 4096; // N-point fft

	FFTData<float> data, cufftData;
	FFTData<double> cufftDataDouble;
	cufftDataDouble.init(N, InitType::eGradient);
	data = cufftData = cufftDataDouble;

	data.upload();
	cufftData.upload();
	cufftDataDouble.upload();

	// custom fft
	perform_custom_fft<N>(data);

	// cuFFT
	perform_cufft(cufftData);

	// cuFFT double
	perform_cufft_double(cufftDataDouble);

	data.download();
	cufftData.download();
	cufftDataDouble.download();

	// output prints
	if (false)
	{
		data.print();
		//cufftData.print();
		//cufftDataDouble.print();
	}

	// comparisons
	if (true)
	{
		printf("\ncustom vs cuFFT:\n");
		compare_fft(data, cufftData);

		printf("\ncustom vs cuFFT double precision:\n");
		compare_fft(data, cufftDataDouble);

		printf("\ncuFFT vs cuFFT double precision:\n");
		compare_fft(cufftData, cufftDataDouble);
	}
}