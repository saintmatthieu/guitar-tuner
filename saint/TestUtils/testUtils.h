#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <optional>
#include <vector>

#include "PitchDetector.h"

namespace saint {
namespace testUtils {

struct Audio {
    Audio(std::vector<float> data, int rate, ChannelFormat format)
        : interleaved(std::move(data)), sampleRate(rate), channelFormat(format) {}
    std::vector<float> interleaved;
    const int sampleRate;
    const ChannelFormat channelFormat;
    int numFrames() const {
        const auto numChannels = channelFormat == ChannelFormat::Mono ? 1 : 2;
        return static_cast<int>(interleaved.size() / numChannels);
    }
};

struct Truth {
    double startTime;
    double endTime;
    float frequency;
};

struct Sample {
    std::filesystem::path file;
    Truth truth;
};

struct ProcessEstimate {
    ProcessEstimate(bool truth, float score, float frequencyEstimate, float rawFrequencyEstimate)
        : t(truth), s(score), f(frequencyEstimate), r(rawFrequencyEstimate) {}
    bool t = false;
    float s = 0.0;
    float f = 0.f;
    float r = 0.f;
};

class TeeStream {
   public:
    TeeStream(std::ostream& console, std::ostream& file) : console_(console), file_(file) {}

    template <typename T>
    TeeStream& operator<<(const T& value) {
        console_ << value;
        file_ << value;
        return *this;
    }

    TeeStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        console_ << manip;
        file_ << manip;
        return *this;
    }

   private:
    std::ostream& console_;
    std::ostream& file_;
};

std::optional<Audio> fromWavFile(std::filesystem::path path, int numSamples = 0);
bool toWavFile(std::filesystem::path path, const Audio& audio, TeeStream* logger,
               const std::string& what = "");

std::filesystem::path getEvalDir();
std::filesystem::path getOutDir();

// Audio processing utilities
void scaleToRms(std::vector<float>& data, float targetRmsDb);
void scaleToPeak(std::vector<float>& data, float targetPeakDb);
void mixNoise(Audio& signal, const std::vector<float>& noise);

// Music theory utilities
float midiNoteToFrequency(int midiNote);
float getTrueFrequency(const std::filesystem::path& filePath);

// File utilities
std::filesystem::path getFileShortName(const std::filesystem::path& filePath);
std::optional<Sample> getSampleFromFile(const std::filesystem::path& filePath);

// Output utilities
struct Marking {
    const int startSample;
    const int endSample;
};
void writeLogMarks(const std::filesystem::path& filenameStem, int sampleRate, Marking marking);

struct Cents {
    float rms = 0.f;
    float avg = 0.f;
};
std::optional<Cents> getError(const Sample& sample, const std::vector<ProcessEstimate>& results);

// Value comparison utility
template <typename T>
bool valueIsUnchanged(const std::filesystem::path& filePath, T previousValue, T newValue,
                      T tolerance = 0) {
    constexpr auto precision = std::numeric_limits<double>::digits10 + 1;
    const auto hasChanged = std::abs(newValue - previousValue) > tolerance;
    if (hasChanged) {
        std::ofstream file{filePath};
        file << std::setprecision(precision) << newValue;
    }
    return !hasChanged;
}

struct RocInfo {
    const double areaUnderCurve;
    const double threshold;
    const std::vector<double> truePositiveRates;
    const std::vector<double> falsePositiveRates;
};

/*!
 * The Receiver Operating Characteristic (ROC) curve is a plot of the true
 * positive rate (TPR) against the false positive rate (FPR) for the different
 * possible thresholds of a binary classifier. The area under the curve (AUC)
 * is a measure of the classifier's performance. The greater the AUC, the
 * better the classifier.
 *
 * @tparam Result has public members `t` (for truth), boolean, and `s` (for score), numeric
 * @param results true classifications and scores of some population
 * @pre at least one of `results` is really positive (`truth` is true), and at
 * least one is really negative
 * @pre `0. <= allowedFalsePositiveRate && allowedFalsePositiveRate <= 1.`
 */
template <typename Result>
RocInfo GetRocInfo(std::vector<Result> results, double allowedFalsePositiveRate = 0.) {
    const auto truth = std::mem_fn(&Result::t);
    const auto falsity = std::not_fn(truth);

    // There is at least one positive and one negative sample.
    assert(any_of(results.begin(), results.end(), truth));
    assert(any_of(results.begin(), results.end(), falsity));

    assert(allowedFalsePositiveRate >= 0. && allowedFalsePositiveRate <= 1.);
    allowedFalsePositiveRate = std::clamp(allowedFalsePositiveRate, 0., 1.);

    // Sort the results by score, descending.
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.s > b.s; });

    const auto size = results.size();
    const auto numPositives = count_if(results.begin(), results.end(), truth);
    const auto numNegatives = size - numPositives;

    // Find true and false positive rates for various score thresholds.
    // True positive and false positive counts are nondecreasing with i,
    // therefore if false positive rate has increased at some i, true positive
    // rate has not decreased.
    std::vector<double> truePositiveRates;
    truePositiveRates.reserve(size);
    std::vector<double> falsePositiveRates;
    falsePositiveRates.reserve(size);
    size_t numTruePositives = 0;
    size_t numFalsePositives = 0;
    for (const auto& result : results) {
        if (result.t)
            ++numTruePositives;
        else
            ++numFalsePositives;
        truePositiveRates.push_back(static_cast<double>(numTruePositives) / numPositives);
        falsePositiveRates.push_back(static_cast<double>(numFalsePositives) / numNegatives);
    }

    // Now find the area under the non-decreasing curve with FPR as x-axis,
    // TPR as y, and i as a parameter.  (This curve is within a square with unit
    // side.)
    double auc = 0.;
    for (size_t i = 0; i <= size; ++i) {
        const auto leftFpr = i == 0 ? 0. : falsePositiveRates[i - 1];
        const auto rightFpr = i == size ? 1. : falsePositiveRates[i];
        const auto leftTpr = i == 0 ? 0. : truePositiveRates[i - 1];
        const auto rightTpr = i == size ? 1. : truePositiveRates[i];
        const auto trapezoid = (rightTpr + leftTpr) * (rightFpr - leftFpr) / 2.;
        assert(trapezoid >= 0);  // See comments above
        auc += trapezoid;
    }

    // Find the parameter at which the x coordinate exceeds the allowed FPR.
    const auto it = std::upper_bound(falsePositiveRates.begin(), falsePositiveRates.end(),
                                     allowedFalsePositiveRate);

    if (it == falsePositiveRates.end())
        // All breakpoints satify the constraint. Return the least score.
        return {auc, results.back().s};
    else if (it == falsePositiveRates.begin())
        // No breakpoint satisfies the constraint. Return the greatest score.
        return {auc, results.front().s};

    // For threshold, use the score halfway between the last breakpoint that
    // satisfies the constraint and the first breakpoint that doesn't.
    const auto index = it - falsePositiveRates.begin();
    const auto threshold = (results[index - 1].s + results[index].s) / 2;

    return {auc, threshold, truePositiveRates, falsePositiveRates};
}

template <typename T>
void PrintPythonVector(std::ofstream& ofs, const std::vector<T>& v, const char* name) {
    ofs << name << " = [";
    std::for_each(v.begin(), v.end(), [&](T x) { ofs << x << ","; });
    ofs << "]\n";
}

struct FileWriter {
    virtual ~FileWriter() = default;
    virtual bool toWavFile(std::filesystem::path path, const Audio& audio, TeeStream* logger,
                           const std::string& what = "") const = 0;
};

struct RealFileWriter : public FileWriter {
    bool toWavFile(std::filesystem::path path, const Audio& audio, TeeStream* logger,
                   const std::string& what = "") const override {
        return testUtils::toWavFile(path, audio, logger, what);
    }
};

struct DummyFileWriter : public FileWriter {
    bool toWavFile(std::filesystem::path path, const Audio& audio, TeeStream* logger,
                   const std::string& what = "") const override {
        return true;
    }
};

}  // namespace testUtils
}  // namespace saint