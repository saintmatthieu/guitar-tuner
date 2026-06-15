#include "PitchDetectorImpl.h"

#include <cmath>

#include "PitchDetectorLoggerInterface.h"

namespace {
// Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
double betaFunction(double a, double b) {
    return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
}

// Beta probability density function
double betaPdf(double x, double a, double b) {
    if (x <= 0.0 || x >= 1.0) {
        return 0.0;
    }
    return std::pow(x, a - 1.0) * std::pow(1.0 - x, b - 1.0) / betaFunction(a, b);
}

// Standard normal PDF: phi(x) = exp(-x^2 / 2) / sqrt(2 * pi)
double standardNormalPdf(double x) {
    constexpr double kInvSqrt2Pi = 0.3989422804014327;
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

// Standard normal CDF approximation using the error function
double standardNormalCdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Skewed normal PDF: f(x; a, loc, scale) = (2/scale) * phi(z) * Phi(a * z)
// where z = (x - loc) / scale
double skewedNormalPdf(double x, double a, double loc, double scale) {
    const double z = (x - loc) / scale;
    return (2.0 / scale) * standardNormalPdf(z) * standardNormalCdf(a * z);
}

// Fitted distributions used to estimate P(correct octave | presence score),
// together with the acceptance thresholds on that probability (with and without
// an active estimate constraint).
struct OctaveModel {
    double betaA;       // good (|err|<=50c): Beta(a, b) on [0, 1]
    double betaB;
    double skewA;       // octaviated: skew-normal(a, loc, scale)
    double skewLoc;
    double skewScale;
    double priorGood;   // mixture weight of the good class
    double threshold;            // accept when P(good) >= this (no constraint)
    double thresholdConstrained; // ... and this when an estimate constraint is active
};

// Fitted on the original (non-noise-compensated) presence-score distribution.
constexpr OctaveModel kBaselineModel{3.388008757503728,  0.4029325165967037, 4.583734827154467,
                                     0.12563587985166158, 0.364240265091698,  0.5911103997932017,
                                     0.85,                0.7};

// Re-fitted on the noise-compensated distribution: subtracting the noise floor
// inflates the presence scores of both classes, so the good/bad models shift
// (and the effective raw-score gate rises). Regenerate with
// eval/fitAndShowErrorProbabilityModels.py on data captured with compensation
// on. The thresholds are re-tuned for this model's score distribution.
constexpr OctaveModel kNoiseCompensatedModel{4.403472452295939,   0.38299747425045966,
                                             1.5009072813805049,  0.27308653270774064,
                                             0.33556510698802494, 0.5547480525652233,
                                             0.85,                0.7};

// Probability of xcorrEstimate not being octaviated given presence score s.
// Uses Bayes' theorem with fitted distributions.
double probabilityNotOctaviated(double s, const OctaveModel& model) {
    // f_(S|G)(s|good) - likelihood of s given good estimate
    const double likelihoodGood = betaPdf(s, model.betaA, model.betaB);

    // f_(S|G)(s|not good) - likelihood of s given octaviated estimate
    const double likelihoodNotGood = skewedNormalPdf(s, model.skewA, model.skewLoc, model.skewScale);

    // f_S(s) - marginal probability (mixture)
    const double marginal =
        model.priorGood * likelihoodGood + (1. - model.priorGood) * likelihoodNotGood;

    if (marginal <= 0.0) {
        return 0.0;
    }

    // P(good|s) = f_(S|G)(s|good) * P(good) / f_S(s)
    return (likelihoodGood * model.priorGood) / marginal;
}
}  // namespace

namespace saint {
PitchDetectorImpl::PitchDetectorImpl(std::unique_ptr<Preprocessor> preprocessor,
                                     FrequencyDomainTransformer transformer,
                                     AutocorrPitchDetector autocorrPitchDetector,
                                     AutocorrEstimateDisambiguator disambiguator,
                                     OnsetDetector onsetDetector,
                                     std::unique_ptr<PitchDetectorLoggerInterface> logger)
    : _preprocessor(std::move(preprocessor)),
      _frequencyDomainTransformer(std::move(transformer)),
      _autocorrPitchDetector(std::move(autocorrPitchDetector)),
      _disambiguator(std::move(disambiguator)),
      _onsetDetector(std::move(onsetDetector)),
      _logger(std::move(logger)) {}

float PitchDetectorImpl::process(const float* audio, DebugOutput* debugOutput,
                                 std::vector<float>* debugOutputSignal) {
    _logger->StartNewEstimate();
    utils::Finally finally{[this] { _logger->EndNewEstimate(nullptr, 0); }};

    // Use the unprocessed, broadband audio for the onset detection.
    const auto isOnset = _onsetDetector.process(audio, debugOutput);
    if (debugOutput) {
        (*debugOutput)["isOnset"] = isOnset ? 1.f : 0.f;
    }
    if (isOnset) {
        // New attack is detected, likely a new note ; reset constraint
        _estimateConstraint.reset();
        // A note is now sounding, so the audio is no longer reliably noise-only.
        // Hold the noise-power estimate at whatever was learned from the
        // pre-onset audio and stop adapting it (see freezeNoiseEstimation).
        _autocorrPitchDetector.freezeNoiseEstimation();
    }

    const auto processedAudio = _preprocessor->processBlock(audio);
    if (debugOutputSignal) {
        debugOutputSignal->insert(debugOutputSignal->end(), processedAudio.begin(),
                                  processedAudio.end());
    }

    const std::vector<std::complex<float>> freq =
        _frequencyDomainTransformer.process(processedAudio.data());

    auto presenceScore = 0.f;
    // If noise compensation is active, `compensatedFreq` receives the
    // noise-subtracted spectrum used for the autocorrelation; we then feed the
    // same denoised spectrum to the octave disambiguator so its harmonic fit
    // sees the cleaned-up partials rather than the original noisy ones.
    std::vector<std::complex<float>> compensatedFreq;
    const float xcorrEstimate = _autocorrPitchDetector.process(freq, &presenceScore,
                                                               _estimateConstraint, &compensatedFreq);
    if (debugOutput) {
        (*debugOutput)["presenceScore"] = presenceScore;
    }

    if (xcorrEstimate == 0.f) {
        return 0.f;
    }

    // Evaluate P(xcorrEstimate is the correct octave | presence score) via Bayes'
    // theorem (see probabilityNotOctaviated / OctaveModel above) and gate on it.
    // The noise-compensated spectrum shifts the presence-score distribution, so
    // it uses its own re-fitted octave model; the unmodified path keeps the
    // original calibration.
    const auto& octaveModel = _autocorrPitchDetector.noiseCompensationEnabled()
                                  ? kNoiseCompensatedModel
                                  : kBaselineModel;
    const double probNotOctaviated = probabilityNotOctaviated(presenceScore, octaveModel);

    // At the time of writing, achieves 99% of estimates within +/-50 cents of the ground truth
    // and 8% of the test cases failing by no-pitch-detected.
    const auto threshold = _estimateConstraint.has_value() ? octaveModel.thresholdConstrained
                                                           : octaveModel.threshold;
    if (probNotOctaviated < threshold) {
        return 0.f;
    }

    const std::vector<std::complex<float>>& spectrumForDisambiguation =
        compensatedFreq.empty() ? freq : compensatedFreq;
    std::vector<float> powerSpectrum;
    utils::getPowerSpectrum(spectrumForDisambiguation, powerSpectrum);
    std::vector<float> dbSpectrum = powerSpectrum;
    std::transform(dbSpectrum.begin(), dbSpectrum.end(), dbSpectrum.begin(),
                   [](float power) { return utils::FastDb(power); });
    assert(utils::isSymmetric(dbSpectrum));
    _logger->Log(dbSpectrum.data(), dbSpectrum.size(), "dbSpectrum");

    const auto disambiguatedEstimate =
        _disambiguator.process(xcorrEstimate, dbSpectrum, _estimateConstraint);

    return disambiguatedEstimate;
}
}  // namespace saint
