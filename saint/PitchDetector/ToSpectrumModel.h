
#include "CommonTypes.h"
#include "Utils.h"

namespace saint {

template <WindowType W>
std::vector<PeakModel> toSpectrumModel(const std::vector<float>& dbSpectrum,
                                       const std::vector<float>& whitenedSpectrum,
                                       double binResolution, double binFrequency,
                                       const std::vector<std::complex<float>>& spectrum,
                                       std::vector<std::complex<float>>& denoisedSpectrum,
                                       std::vector<float>* fullIdeal = nullptr) {
    assert(utils::isSymmetric(dbSpectrum));
    assert(utils::isSymmetric(whitenedSpectrum));
    assert(dbSpectrum.size() == whitenedSpectrum.size());

    const auto N = dbSpectrum.size() / 2;
    constexpr auto maxNumPeaks = 20;
    std::vector<PeakModel> spectrumModel;
    spectrumModel.reserve(maxNumPeaks);
    if (fullIdeal) {
        fullIdeal->resize(N);
        std::fill(fullIdeal->begin(), fullIdeal->end(), -1000.f);
    }

    denoisedSpectrum.resize(spectrum.size());
    std::fill(denoisedSpectrum.begin(), denoisedSpectrum.end(), std::complex<float>(0.f, 0.f));

    const auto numBinsForFit =
        static_cast<int>(utils::mainLobeWidth<W>() / binResolution / 2) * 2 + 1;
    std::vector<float> idealLobeScratch(numBinsForFit);
    std::vector<float> lobeFittingWeights(numBinsForFit);

    std::vector<int> peakIndices;
    float largestPeakValue = -std::numeric_limits<float>::infinity();
    for (auto i = 1; i < N - 1; ++i) {
        if (dbSpectrum[i] > dbSpectrum[i - 1] && dbSpectrum[i] > dbSpectrum[i + 1] &&
            whitenedSpectrum[i] > 0.f) {
            peakIndices.push_back(i);
            if (dbSpectrum[i] > largestPeakValue) {
                largestPeakValue = dbSpectrum[i];
            }
        }
    }

    {
        // Perceptual masking
        constexpr auto dbPerBark = 20.f;
        std::vector<int> indicesToRemove;
        const auto numIndices = static_cast<int>(peakIndices.size());
        for (int i = 0; i < numIndices; ++i) {
            const auto pi = peakIndices[i];
            const auto thisLevel = dbSpectrum[pi];
            const auto thisBark = utils::toBark(pi, binFrequency);

            const auto maxBarkDiff = (largestPeakValue - thisLevel) / dbPerBark;

            bool masked = false;
            for (int j = i + 1; j < numIndices && !masked; ++j) {
                const auto pj = peakIndices[j];
                const auto otherLevel = dbSpectrum[pj];
                if (otherLevel <= thisLevel)
                    continue;
                const auto barkDiff = std::abs(thisBark - utils::toBark(pj, binFrequency));
                if (barkDiff > maxBarkDiff)
                    break;
                if (thisLevel < otherLevel - dbPerBark * barkDiff) {
                    masked = true;
                    break;
                }
            }

            if (!masked) {
                for (int j = i - 1; j >= 0 && !masked; --j) {
                    const auto pj = peakIndices[j];
                    const auto otherLevel = dbSpectrum[pj];
                    if (otherLevel <= thisLevel)
                        continue;
                    const auto barkDiff = std::abs(thisBark - utils::toBark(pj, binFrequency));
                    if (barkDiff > maxBarkDiff)
                        break;
                    if (thisLevel < otherLevel - dbPerBark * barkDiff) {
                        masked = true;
                        break;
                    }
                }
            }

            if (masked)
                indicesToRemove.push_back(static_cast<int>(i));
        }
        for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it)
            peakIndices.erase(peakIndices.begin() + *it);
    }

    // 0. Create a vector of indices [0, 1, ..., N-1].
    // 1. Find `dbSpectrum`'s global max.
    // 2. Find the troughs left and right of it.
    // 3. Use a quadratic fit to refine the index estimate of the peak.
    // 4. Evaluate the ideal lobe at the fractional bins of the peak.
    // 5. Calculate the mean of the square errors between ideal and actual peak.
    // 6. If the error is larger than a threshold (to be found and tuned), finish.
    // 7. Add an entry to `spectrumModel`.
    // 8. Erase dbSpectrum and index vector entries of the peak.
    // 9. Go back to 1.

    // 0.
    while (!peakIndices.empty() && spectrumModel.size() <= maxNumPeaks) {
        // 1.
        const auto maxIt =
            std::max_element(peakIndices.begin(), peakIndices.end(),
                             [&dbSpectrum](int a, int b) { return dbSpectrum[a] < dbSpectrum[b]; });
        const auto maxIndex = *maxIt;
        const auto maxValue = dbSpectrum[maxIndex];
        peakIndices.erase(maxIt);

        // 2.
        const auto beginIndex = maxIndex - (numBinsForFit - 1) / 2;
        const auto endIndex = beginIndex + numBinsForFit;
        if (beginIndex < 0 || endIndex > N) {
            break;
        }

        // 3.
        float refinedIndex = maxIndex;
        float refinedValue = maxValue;
        if (maxIndex > 0 && maxIndex < N - 1) {
            const float y[3] = {dbSpectrum[maxIndex - 1], dbSpectrum[maxIndex],
                                dbSpectrum[maxIndex + 1]};
            refinedIndex += utils::quadFit(y, &refinedValue);
        }

        // 4.
        for (auto i = beginIndex; i < endIndex; ++i) {
            const auto linearValue = utils::mainLobeAt<W>((i - refinedIndex) * binResolution);
            const auto db = utils::FastDb(linearValue * linearValue) + refinedValue;
            idealLobeScratch[i - beginIndex] = db;
            lobeFittingWeights[i - beginIndex] =
                1 - std::pow((i - refinedIndex) * 2 / numBinsForFit, 16);
        }
        if (fullIdeal) {
            for (auto i = beginIndex; i < endIndex; ++i) {
                (*fullIdeal)[i] = idealLobeScratch[i - beginIndex];
            }
        }

        // 5.
        const std::vector<float> actualLobe{dbSpectrum.begin() + beginIndex,
                                            dbSpectrum.begin() + endIndex};
        auto squareSum = 0.f;
        for (auto i = 0; i < numBinsForFit; ++i) {
            const auto error = idealLobeScratch[i] - actualLobe[i];
            squareSum += error * error * lobeFittingWeights[i];
        }
        const auto rmsError = std::sqrt(squareSum / numBinsForFit);

        // 6.
        constexpr auto coef = -0.5157894736842106f;
        const auto weight = rmsError < -coef ? 1.f : std::min(-coef / (coef + rmsError), 1.f);
        if (weight > 0.1) {
            spectrumModel.push_back({refinedIndex, refinedValue, weight});
            std::copy(spectrum.begin() + beginIndex, spectrum.begin() + endIndex,
                      denoisedSpectrum.begin() + beginIndex);
        }
    }

    // sort peaks by index
    std::sort(spectrumModel.begin(), spectrumModel.end(),
              [](const PeakModel& a, const PeakModel& b) { return a.index < b.index; });

    return spectrumModel;
}

}  // namespace saint