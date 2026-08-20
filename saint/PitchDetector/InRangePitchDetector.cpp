#include "InRangePitchDetector.h"

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

// Probability of xcorrEstimate not being octaviated given presence score s.
// Uses Bayes' theorem with fitted distributions.
double probabilityNotOctaviated(double s) {
    // Distribution parameters, fitted by eval/fitAndShowErrorProbabilityModels.py from
    // raw per-frame presence/error pairs (collected with disableOctaviationGate=true
    // testWithMedianFilter=false). The fit is specific to the averaging setting, so it
    // must be regenerated whenever autocorrAveragingFrameCount changes.
    //
    // Active set: autocorrAveragingFrameCount=4 (cross-frame ACF averaging).
    constexpr double kBetaA = 2.7634801603149746;
    constexpr double kBetaB = 0.39897836666380115;
    constexpr double kSkewA = 6.89477379633251;
    constexpr double kSkewLoc = 0.07457877837828505;
    constexpr double kSkewScale = 0.3471294772559438;
    constexpr double kPriorGood = 0.5733624834956706;
    constexpr double kPriorNotGood = 1. - kPriorGood;
    // For autocorrAveragingFrameCount=1 (no averaging), use:
    //   kBetaA=3.388008757503728, kBetaB=0.4029325165967037, kSkewA=4.583734827154467,
    //   kSkewLoc=0.12563587985166158, kSkewScale=0.364240265091698, kPriorGood=0.5911103997932017

    // f_(S|G)(s|good) - likelihood of s given good estimate
    const double likelihoodGood = betaPdf(s, kBetaA, kBetaB);

    // f_(S|G)(s|not good) - likelihood of s given octaviated estimate
    const double likelihoodNotGood = skewedNormalPdf(s, kSkewA, kSkewLoc, kSkewScale);

    // f_S(s) - marginal probability (mixture)
    const double marginal = kPriorGood * likelihoodGood + kPriorNotGood * likelihoodNotGood;

    if (marginal <= 0.0) {
        return 0.0;
    }

    // P(good|s) = f_(S|G)(s|good) * P(good) / f_S(s)
    return (likelihoodGood * kPriorGood) / marginal;
}
}  // namespace

namespace saint {
InRangePitchDetector::InRangePitchDetector(FrequencyDomainTransformer transformer,
                                           AutocorrPitchDetector autocorrPitchDetector,
                                           AutocorrEstimateDisambiguator disambiguator,
                                           PitchDetectorLoggerInterface& logger,
                                           OctaviationGateConfig gate)
    : _frequencyDomainTransformer(std::move(transformer)),
      _autocorrPitchDetector(std::move(autocorrPitchDetector)),
      _disambiguator(std::move(disambiguator)),
      _logger(logger),
      _applyOctaviationGate(gate.apply),
      _presenceThreshold(gate.presenceThreshold),
      _harmonicityFloor(gate.harmonicityFloor),
      _presenceThresholdWithConstraint(gate.presenceThresholdWithConstraint) {}

void InRangePitchDetector::onNewOnset() {
    _estimateConstraint.reset();
    _autocorrPitchDetector.reset();
}

PitchDetectionResult InRangePitchDetector::process(const std::vector<float>& audio,
                                                   DebugOutput* debugOutput) {
    const std::vector<std::complex<float>>& freq = _frequencyDomainTransformer.process(audio);

    auto presenceScore = 0.f;
    const float xcorrEstimate =
        _autocorrPitchDetector.process(freq, &presenceScore, _estimateConstraint);
    if (debugOutput) {
        (*debugOutput)["presenceScore"] = presenceScore;
        (*debugOutput)["xcorrEstimate"] = xcorrEstimate;
    }

    if (xcorrEstimate == 0.f) {
        return {};
    }

    // Evaluate the probability of xcorrEstimate not being octaviated ("good") given
    // the presence score "s": P(good|s) = f(s|good) P(good) / f_S(s), where f(s|good)
    // is a Beta pdf, f(s|not good) a skewed-normal pdf, and f_S(s) their prior-weighted
    // mixture. The fitted parameters live in probabilityNotOctaviated() above and are
    // produced by eval/fitAndShowErrorProbabilityModels.py from errors.py.
    const double probNotOctaviated = probabilityNotOctaviated(presenceScore);
    if (debugOutput) {
        // Reuse a static key: this label exceeds libstdc++'s 15-char small-string buffer, so a
        // fresh temporary would heap-allocate on every block (the production median filter always
        // passes a debug map). With an existing key in a reused map, operator[] then allocates
        // nothing.
        static const std::string kProbNotOctaviated = "probNotOctaviated";
        (*debugOutput)[kProbNotOctaviated] = static_cast<float>(probNotOctaviated);
    }

    // The spectrum and disambiguation are computed before the gate so the gate can
    // also weigh the harmonicity of the octave-corrected estimate (#4).
    utils::getPowerSpectrum(freq, _dbSpectrum);
    std::transform(_dbSpectrum.begin(), _dbSpectrum.end(), _dbSpectrum.begin(),
                   [](float power) { return utils::FastDb(power); });
    assert(utils::isSymmetric(_dbSpectrum));
    _logger.Log(_dbSpectrum.data(), _dbSpectrum.size(), "dbSpectrum");

    float harmonicity = 0.f;
    const auto disambiguatedEstimate =
        _disambiguator.process(xcorrEstimate, _dbSpectrum, _estimateConstraint,
                               // Don't rely on harmonicity once we've locked: it'll be low once the
                               // high-order partials have decayed and then it'll be useless.
                               _estimateConstraint.has_value() ? nullptr : &harmonicity);
    if (debugOutput) {
        (*debugOutput)["harmonicity"] = harmonicity;
    }

    // Gate: reject when the presence-based octaviation probability is too low, or the
    // estimate lacks harmonic support. Once locked, the search/disambiguation are already
    // clamped to a major third of the constraint, so this cut is a pure presence gate and
    // is set more permissively (it governs how long a decaying note keeps being tracked).
    // harmonicityFloor=0 disables the harmonic criterion.
    const auto threshold =
        _estimateConstraint.has_value() ? _presenceThresholdWithConstraint : _presenceThreshold;

    auto harmonicityReject = false;
    if (!_estimateConstraint.has_value()) {
        harmonicityReject = harmonicity < _harmonicityFloor;
    }

    const auto gateRejects =
        _applyOctaviationGate && (probNotOctaviated < threshold || harmonicityReject);

    if (gateRejects) {
        return {};
    }

    return {disambiguatedEstimate, PitchBucket::inRange};
}

}  // namespace saint
