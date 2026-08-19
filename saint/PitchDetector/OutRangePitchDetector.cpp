#include "OutRangePitchDetector.h"

#include "PitchDetectorLoggerInterface.h"
#include "Utils.h"

namespace saint {
OutRangePitchDetector::OutRangePitchDetector(std::unique_ptr<Preprocessor> preprocessor,
                                             OnsetDetector onsetDetector,
                                             std::unique_ptr<LowBandAnalyzer> lowBandAnalyzer,
                                             std::unique_ptr<InRangePitchDetector> inRangeDetector,
                                             std::unique_ptr<PitchDetectorLoggerInterface> logger,
                                             int decimationFactor, LowBandConfig lowBand,
                                             OutRangeConfig outRange)
    : _logger(std::move(logger)),
      _preprocessor(std::move(preprocessor)),
      _onsetDetector(std::move(onsetDetector)),
      _lowBandAnalyzer(std::move(lowBandAnalyzer)),
      _inRangeDetector(std::move(inRangeDetector)),
      _decimationFactor(decimationFactor),
      _lowBand(lowBand),
      _presenceThreshold(outRange.presenceThreshold) {}

PitchDetectionResult OutRangePitchDetector::process(const float* input, DebugOutput* debugOutput,
                                                    std::vector<float>* debugOutputSignal) {
    // True on the single frame the logger records (see PitchDetectorLogger): the below-range
    // diagnostics are only assembled then.
    const auto logging = _logger->StartNewEstimate();
    utils::Finally finally{[this] { _logger->EndNewEstimate(nullptr, 0); }};

    if (debugOutput == nullptr) {
        debugOutput = &_debugOutput;
    }
    const auto report = [debugOutput](PitchDetectionResult result) {
        (*debugOutput)["rawBucket"] =
            result.bucket.has_value() ? static_cast<float>(*result.bucket) : -1.f;
        return result;
    };

    // Use the unprocessed, broadband audio for the onset detection.
    const auto isOnset = _onsetDetector.process(input, debugOutput);
    (*debugOutput)["isOnset"] = isOnset ? 1.f : 0.f;
    if (isOnset) {
        _inRangeDetector->onNewOnset();
    }

    const auto& processedAudio = _preprocessor->processBlock(input);
    if (debugOutputSignal) {
        debugOutputSignal->insert(debugOutputSignal->end(), processedAudio.begin(),
                                  processedAudio.end());
    }
    // Fed every block, whether or not this frame ends up asking it anything.
    _lowBandAnalyzer->process(processedAudio);

    const auto inRange = _inRangeDetector->process(processedAudio, debugOutput);
    if (inRange.bucket.has_value() && !likelyLowBand(debugOutput, inRange.pitch, logging)) {
        return report(inRange);
    }

    // Nothing in range, so the question is only whether something is sounding below it - a much
    // easier one than which pitch, and the low band's own autocorrelation answers it on a window
    // long enough for those periods, which the in-range presence score is not.
    const auto presence = _lowBandAnalyzer->presence();
    (*debugOutput)["lowBandPresence"] = presence;
    if (presence < _presenceThreshold) {
        return report({});
    }
    return report({0.f, PitchBucket::belowRange});
}

bool OutRangePitchDetector::likelyLowBand(DebugOutput* debugOutput, float inRangeEstimate,
                                          bool logging) {
    const auto lowBand =
        _lowBandAnalyzer->below(inRangeEstimate, logging ? &_lowBandDiagnostics : nullptr);
    (*debugOutput)["lowBandHz"] = lowBand.frequency;
    (*debugOutput)["lowBandSupport"] = lowBand.support;
    if (logging) {
        logLowBand(lowBand, inRangeEstimate);
    }
    return lowBand.frequency > 0.f && lowBand.support >= _lowBand.harmonicSupportFloor;
}

void OutRangePitchDetector::logLowBand(const LowBandAnalyzer::Verdict& verdict,
                                       float inRangeEstimate) const {
    const auto& d = _lowBandDiagnostics;
    _logger->Log(inRangeEstimate, "lowBandInRangeHz");
    _logger->Log(verdict.frequency, "lowBandVerdictHz");
    _logger->Log(verdict.support, "lowBandVerdictSupport");
    _logger->Log(_lowBand.harmonicSupportFloor, "lowBandSupportFloor");
    _logger->Log(d.candidateHz.data(), d.candidateHz.size(), "lowBandCandidateHz");
    _logger->Log(d.candidateSupport.data(), d.candidateSupport.size(), "lowBandCandidateSupport");
    _logger->Log(d.candidateProminence.data(), d.candidateProminence.size(),
                 "lowBandCandidateProminence");
    _logger->Log(d.combHz.data(), d.combHz.size(), "lowBandCombHz");
    _logger->Log(d.combProminence.data(), d.combProminence.size(), "lowBandCombProminence");
    _logger->Log(d.combExplained.data(), d.combExplained.size(), "lowBandCombExplained");
}

int OutRangePitchDetector::delaySamples() const {
    return _decimationFactor * _inRangeDetector->delaySamples();
}

std::pair<float, float> OutRangePitchDetector::pitchSearchRange() const {
    return _inRangeDetector->pitchSearchRange();
}

void OutRangePitchDetector::setEstimateConstraint(float constraint) {
    _inRangeDetector->setEstimateConstraint(constraint);
}
}  // namespace saint
