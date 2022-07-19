#pragma once

#include <numeric>

// compare using SQNR
static void compare_sqnr(/*uint nComplex,*/ uint nSamples)
{
	static constexpr uint nWords = 8;
	static constexpr uint nComplex = 1024; // currently needed cuz custom fft is templated

	std::vector<FFTData<double>> ideals(nSamples);
	std::vector<FFTData<float>> others(nSamples);

	// calculate ffts
	for (uint i = 0; i < nSamples; i++) 
	{
		// refs for ease of writing
		FFTData<double>& ideal = ideals[i];
		FFTData<float>& other = others[i];

		// init both fft inputs with the same values (-1.0 to 1.0)
		// -> create values as double, cast to float for "other"
		other = ideal.init(nComplex, InitType::eRandom);

		// upload from host to device
		ideal.upload();
		other.upload();

		// execute ffts
		perform_cufft_double(ideal);
		//perform_custom_fft<nComplex>(other);
		perform_cufft(other);

		// download from device to host
		ideal.download();
		other.download();
	}

	// calculate sqnr
	{
		// SQNR = 10*log10 (mean(sum(abs(yIdeal).^2)) / mean(sum(abs(yIdeal - yFFTq).^2)));
		std::vector<double> a(nSamples), b(nSamples);
		//mean
		for (uint iSample = 0; iSample < nSamples; iSample++)
		{
			// sum
			for (uint iComplex = 0; iComplex < nComplex; iComplex++)
			{
				// TODO: do pointer arithmetic instead

				// real
				double ideal = (double)ideals[iSample].vals[iComplex].real;
				double other = (double)others[iSample].vals[iComplex].real;

				a[iSample] += abs(ideal) * abs(ideal);
				b[iSample] += abs(ideal - other) * abs(ideal - other);

				// imag
				ideal = (double)ideals[iSample].vals[iComplex].imag;
				other = (double)others[iSample].vals[iComplex].imag;

				a[iSample] += abs(ideal) * abs(ideal);
				b[iSample] += abs(ideal - other) * abs(ideal - other);
			}
		}
		// TODO: could use std::reduce in parallel exec mode for faster sum
		double a_sum = std::accumulate(a.cbegin(), a.cend(), 0.0);
		double b_sum = std::accumulate(b.cbegin(), b.cend(), 0.0);
		double mean_inv = 1.0 / (double)(nSamples + nComplex * 2);
		double sqnr = 10 * log10((a_sum * mean_inv) / (b_sum * mean_inv));
		printf("SQNR: %.2f\n", sqnr);
	}
}