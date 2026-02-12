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

// Probability of xcorrEstimate not being octaviated given presence score s.
// Uses Bayes' theorem with fitted distributions.
double probabilityNotOctaviated(double s) {
    // Distribution parameters (fitted from data)
    constexpr double kBetaA = 3.388008757503728;
    constexpr double kBetaB = 0.4029325165967037;
    constexpr double kSkewA = 4.583734827154467;
    constexpr double kSkewLoc = 0.12563587985166158;
    constexpr double kSkewScale = 0.364240265091698;
    constexpr double kPriorGood = 0.5911103997932017;
    constexpr double kPriorNotGood = 1. - kPriorGood;

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
PitchDetectorImpl::PitchDetectorImpl(FrequencyDomainTransformer transformer,
                                     AutocorrPitchDetector autocorrPitchDetector,
                                     AutocorrEstimateDisambiguator disambiguator,
                                     std::unique_ptr<PitchDetectorLoggerInterface> logger)
    : _frequencyDomainTransformer(std::move(transformer)),
      _autocorrPitchDetector(std::move(autocorrPitchDetector)),
      _disambiguator(std::move(disambiguator)),
      _logger(std::move(logger)) {}

float PitchDetectorImpl::process(const float* audio, float* outPresenceScore) {
    _logger->StartNewEstimate();
    utils::Finally finally{[this] { _logger->EndNewEstimate(nullptr, 0); }};

    const std::vector<std::complex<float>> freq = _frequencyDomainTransformer.process(audio);

    auto presenceScore = 0.f;
    const float xcorrEstimate =
        _autocorrPitchDetector.process(freq, &presenceScore, _estimateConstraint);
    if (outPresenceScore) {
        *outPresenceScore = presenceScore;
    }

    if (xcorrEstimate == 0.f) {
        return 0.f;
    }

    // clang-format off
    // Evaluate the probability of xcorrEstimate not being octaviated (being "good") given presence
    // score "s".
    // P(good|s) = f_(S|G)(s|good) * P(good) / f_S(s) where
    // f_(S|G)(s|good) is modeled by a beta function of s with parameters
    //      (a, b) = (1.8116072489777812, 0.360943336209587)
    // f_(S|G)(s|not good) is modeled by a skewed normal distribution with parameters
    //      (a, loc, scale) = (4.551395913440457, 0.12565570910063836, 0.3627118137538335)
    // f_S(s) is a mixture of both with weights 0.665 for the "good" distribution and 0.335 for the "not good".
    // clang-format on
    const double probNotOctaviated = probabilityNotOctaviated(presenceScore);

    // At the time of writing, achieves 99% of estimates within +/-50 cents of the ground truth
    // and 8% of the test cases failing by no-pitch-detected.
    constexpr auto threshold = 0.94;
    if (probNotOctaviated < threshold) {
        return 0.f;
    }

    const auto disambiguatedEstimate =
        _disambiguator.process(xcorrEstimate, freq, _estimateConstraint);

    return disambiguatedEstimate;
}
}  // namespace saint
