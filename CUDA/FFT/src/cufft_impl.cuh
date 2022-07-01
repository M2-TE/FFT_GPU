#include <cufft.h>

void perform_cufft(FFTData& data)
{
	cufftHandle plan;
	cufftPlan1d(&plan, data.size(), CUFFT_C2C, 1);

	cufftComplex* inOut = reinterpret_cast<cufftComplex*>(data.deviceData);
	cufftExecC2C(plan, inOut, inOut, CUFFT_FORWARD);

	cufftDestroy(plan);
}