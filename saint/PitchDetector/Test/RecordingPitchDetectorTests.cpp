#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

#include "PitchDetectorFactory.h"
#include "Recording/IRecordingListener.h"
#include "Recording/IssueReportingPitchDetector.h"
#include "Recording/PitchDetectorRecording.h"
#include "Recording/RecordingPitchDetector.h"
#include "Recording/ReplayPitchDetector.h"
#include "testUtils.h"

namespace saint {
namespace {
class StubPitchDetector : public PitchDetector {
   public:
    float process(const float*, DebugOutput*, std::vector<float>*) override {
        return static_cast<float>(++_numCalls);
    }
    int delaySamples() const override {
        return 123;
    }

   private:
    int _numCalls = 0;
};

class StubListener : public IRecordingListener {
   public:
    void onProgress(int remainingSeconds) override {
        progress.push_back(remainingSeconds);
    }
    void onComplete(recording::RecordingData d) override {
        data = std::move(d);
        ++numCompletions;
    }

    std::vector<int> progress;
    std::optional<recording::RecordingData> data;
    int numCompletions = 0;
};

std::vector<float> sineBlocks(int numBlocks, int samplesPerBlock, float freq, int sampleRate) {
    std::vector<float> samples(static_cast<size_t>(numBlocks) * samplesPerBlock);
    constexpr auto twoPi = 6.283185307179586;
    for (size_t n = 0; n < samples.size(); ++n) {
        samples[n] = 0.5f * static_cast<float>(std::sin(twoPi * freq * n / sampleRate));
    }
    return samples;
}

// sampleRate 100, 10 samples per block -> recording capacity of exactly
// stubRecordingSeconds * 10 = 100 blocks.
const recording::PitchDetectorConfig stubConfig{100, ChannelFormat::Mono, 10, Tuning::Standard};
constexpr auto stubRecordingSeconds = 10;
constexpr auto stubCapacityBlocks = stubRecordingSeconds * 10;
}  // namespace

TEST(PitchDetectorRecording, configSerializationRoundTrip) {
    const recording::PitchDetectorConfig config{48000, ChannelFormat::Stereo, 512,
                                                Tuning::Standard};
    const auto deserialized = recording::deserializeConfig(recording::serializeConfig(config));
    ASSERT_TRUE(deserialized.has_value());
    EXPECT_EQ(deserialized->sampleRate, config.sampleRate);
    EXPECT_EQ(deserialized->channelFormat, config.channelFormat);
    EXPECT_EQ(deserialized->samplesPerBlockPerChannel, config.samplesPerBlockPerChannel);
    EXPECT_EQ(deserialized->tuning, config.tuning);
}

TEST(PitchDetectorRecording, deserializeConfigRejectsIncompleteInput) {
    EXPECT_FALSE(recording::deserializeConfig("").has_value());
    EXPECT_FALSE(recording::deserializeConfig("sampleRate=48000").has_value());
    EXPECT_FALSE(
        recording::deserializeConfig(
            "sampleRate=48000;channelFormat=Quadro;samplesPerBlockPerChannel=512;tuning=Standard")
            .has_value());
}

TEST(PitchDetectorRecording, wavFileRoundTrip) {
    const recording::PitchDetectorConfig config{44100, ChannelFormat::Mono, 441, Tuning::Standard};
    std::vector<float> samples(441 * 3);
    std::iota(samples.begin(), samples.end(), 0.f);
    const auto path = testUtils::getOutDir() / "recordingRoundTrip.wav";

    ASSERT_TRUE(recording::writeWavFile(path, config, samples.data(), samples.size()));
    const auto read = recording::readWavFile(path);
    ASSERT_TRUE(read.has_value());
    EXPECT_EQ(read->config.sampleRate, config.sampleRate);
    EXPECT_EQ(read->config.channelFormat, config.channelFormat);
    EXPECT_EQ(read->config.samplesPerBlockPerChannel, config.samplesPerBlockPerChannel);
    EXPECT_EQ(read->config.tuning, config.tuning);
    EXPECT_EQ(read->interleaved, samples);
}

TEST(RecordingPitchDetector, forwardsToInnerDetector) {
    StubListener listener;
    RecordingPitchDetector detector(std::make_unique<StubPitchDetector>(), stubConfig,
                                    stubRecordingSeconds, listener, nullptr);
    const std::vector<float> block(stubConfig.samplesPerBlockPerChannel, 0.f);
    EXPECT_EQ(detector.process(block.data()), 1.f);
    EXPECT_EQ(detector.process(block.data()), 2.f);
    EXPECT_EQ(detector.delaySamples(), 123);
}

TEST(RecordingPitchDetector, reportsTheRemainingSecondsOnceASecond) {
    StubListener listener;
    RecordingPitchDetector detector(std::make_unique<StubPitchDetector>(), stubConfig,
                                    stubRecordingSeconds, listener, nullptr);
    const std::vector<float> block(stubConfig.samplesPerBlockPerChannel, 0.f);
    for (auto i = 0; i < stubCapacityBlocks; ++i) {
        detector.process(block.data());
    }
    EXPECT_EQ(listener.progress, std::vector<int>({10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST(RecordingPitchDetector, handsBackTheInnerDetectorAndTheDataWhenComplete) {
    StubListener listener;
    std::unique_ptr<PitchDetector> handedBack;
    std::optional<recording::RecordingData> recordedData;
    auto numCompletions = 0;
    RecordingPitchDetector detector(
        std::make_unique<StubPitchDetector>(), stubConfig, stubRecordingSeconds, listener,
        [&](std::unique_ptr<PitchDetector> inner, recording::RecordingData data) {
            handedBack = std::move(inner);
            recordedData = std::move(data);
            ++numCompletions;
        });

    std::vector<float> block(stubConfig.samplesPerBlockPerChannel);
    for (auto i = 0; i < stubCapacityBlocks; ++i) {
        EXPECT_EQ(handedBack, nullptr);
        std::fill(block.begin(), block.end(), static_cast<float>(i));
        detector.process(block.data());
    }
    EXPECT_EQ(numCompletions, 1);
    ASSERT_NE(handedBack, nullptr);
    // The handed-back detector continues where the recording left off.
    EXPECT_EQ(handedBack->process(block.data()), stubCapacityBlocks + 1.f);

    // The recorded data round-trips through the client-written WAV file.
    ASSERT_TRUE(recordedData.has_value());
    const auto path = testUtils::getOutDir() / "recordingComplete.wav";
    ASSERT_TRUE(recording::writeWavFile(path, *recordedData));
    const auto read = recording::readWavFile(path);
    ASSERT_TRUE(read.has_value());
    ASSERT_EQ(read->interleaved.size(),
              static_cast<size_t>(stubCapacityBlocks) * stubConfig.samplesPerBlockPerChannel);
    for (auto i = 0; i < stubCapacityBlocks; ++i) {
        EXPECT_EQ(read->interleaved[static_cast<size_t>(i) * stubConfig.samplesPerBlockPerChannel],
                  static_cast<float>(i))
            << "block " << i;
    }
}

TEST(RecordingPitchDetector, stopTerminatesEarlyWithTheBlocksRecordedSoFar) {
    StubListener listener;
    std::optional<recording::RecordingData> recordedData;
    RecordingPitchDetector detector(
        std::make_unique<StubPitchDetector>(), stubConfig, stubRecordingSeconds, listener,
        [&](std::unique_ptr<PitchDetector>, recording::RecordingData data) {
            recordedData = std::move(data);
        });

    const std::vector<float> block(stubConfig.samplesPerBlockPerChannel, 0.f);
    for (auto i = 0; i < 5; ++i) {
        detector.process(block.data());
    }
    detector.stop();
    ASSERT_TRUE(recordedData.has_value());
    EXPECT_EQ(recordedData->interleaved.size(), 5u * stubConfig.samplesPerBlockPerChannel);
    // Stopping again is a no-op.
    recordedData.reset();
    detector.stop();
    EXPECT_FALSE(recordedData.has_value());
}

TEST(IssueReportingPitchDetector, exchangesTheDetectorOnStartAndKeepsItOnCompletion) {
    auto numDetectorsCreated = 0;
    IssueReportingPitchDetector detector(stubConfig, [&] {
        ++numDetectorsCreated;
        return std::make_unique<StubPitchDetector>();
    });
    EXPECT_EQ(numDetectorsCreated, 1);

    const std::vector<float> block(stubConfig.samplesPerBlockPerChannel, 0.f);
    EXPECT_EQ(detector.process(block.data()), 1.f);
    EXPECT_EQ(detector.process(block.data()), 2.f);

    StubListener listener;
    EXPECT_FALSE(detector.isRecording());
    detector.startIssueRecording(stubRecordingSeconds, listener);
    EXPECT_TRUE(detector.isRecording());
    // A fresh detector was created for the recording: the estimates restart from 1.
    EXPECT_EQ(numDetectorsCreated, 2);
    for (auto i = 0; i < stubCapacityBlocks; ++i) {
        EXPECT_EQ(detector.process(block.data()), i + 1.f) << "block " << i;
    }
    EXPECT_EQ(listener.numCompletions, 1);
    ASSERT_TRUE(listener.data.has_value());
    EXPECT_EQ(listener.data->interleaved.size(),
              static_cast<size_t>(stubCapacityBlocks) * stubConfig.samplesPerBlockPerChannel);
    EXPECT_FALSE(detector.isRecording());
    // Completing the recording does not reset the state: the wrapper took ownership of the
    // recorder's detector and continues feeding it.
    EXPECT_EQ(detector.process(block.data()), stubCapacityBlocks + 1.f);
    EXPECT_EQ(numDetectorsCreated, 2);
}

TEST(IssueReportingPitchDetector, restartsOnTheFlyWhenStartIsCalledWhileRecording) {
    auto numDetectorsCreated = 0;
    IssueReportingPitchDetector detector(stubConfig, [&] {
        ++numDetectorsCreated;
        return std::make_unique<StubPitchDetector>();
    });
    const std::vector<float> block(stubConfig.samplesPerBlockPerChannel, 0.f);

    StubListener firstListener;
    detector.startIssueRecording(stubRecordingSeconds, firstListener);
    for (auto i = 0; i < 5; ++i) {
        detector.process(block.data());
    }
    EXPECT_FALSE(firstListener.data.has_value());

    StubListener secondListener;
    detector.startIssueRecording(stubRecordingSeconds, secondListener);
    // The first recording was gracefully terminated, handing over the blocks recorded so far.
    ASSERT_TRUE(firstListener.data.has_value());
    EXPECT_EQ(firstListener.data->interleaved.size(),
              5u * stubConfig.samplesPerBlockPerChannel);
    EXPECT_TRUE(detector.isRecording());
    EXPECT_EQ(numDetectorsCreated, 3);

    for (auto i = 0; i < stubCapacityBlocks; ++i) {
        EXPECT_EQ(detector.process(block.data()), i + 1.f) << "block " << i;
    }
    ASSERT_TRUE(secondListener.data.has_value());
    EXPECT_EQ(secondListener.data->interleaved.size(),
              static_cast<size_t>(stubCapacityBlocks) * stubConfig.samplesPerBlockPerChannel);
    EXPECT_FALSE(detector.isRecording());
}

TEST(IssueReportingPitchDetector, issueRecordingIsReplayableBitExactly) {
    constexpr auto sampleRate = 44100;
    constexpr auto samplesPerBlock = 441;
    constexpr auto recordingSeconds = 3;
    constexpr auto capacityBlocks = recordingSeconds * sampleRate / samplesPerBlock;

    const auto detector =
        PitchDetectorFactory::createInstance(sampleRate, ChannelFormat::Mono, samplesPerBlock);
    ASSERT_NE(detector, nullptr);

    const auto samples = sineBlocks(capacityBlocks + 10, samplesPerBlock, 110.f, sampleRate);
    // A few blocks before the recording starts, so that the detector state at recording start
    // differs from the initial state.
    for (auto i = 0; i < 10; ++i) {
        detector->process(samples.data() + static_cast<size_t>(i) * samplesPerBlock);
    }

    StubListener listener;
    detector->startIssueRecording(recordingSeconds, listener);
    std::vector<float> liveEstimates;
    for (auto i = 10; i < capacityBlocks + 10; ++i) {
        liveEstimates.push_back(
            detector->process(samples.data() + static_cast<size_t>(i) * samplesPerBlock));
    }
    ASSERT_TRUE(listener.data.has_value());
    const auto path = testUtils::getOutDir() / "issueReportReplay.wav";
    ASSERT_TRUE(recording::writeWavFile(path, *listener.data));

    const auto replayDetector = ReplayPitchDetector::fromFile(path);
    ASSERT_NE(replayDetector, nullptr);
    EXPECT_EQ(replayDetector->config().sampleRate, sampleRate);
    EXPECT_EQ(replayDetector->config().channelFormat, ChannelFormat::Mono);
    EXPECT_EQ(replayDetector->config().samplesPerBlockPerChannel, samplesPerBlock);
    EXPECT_EQ(replayDetector->config().tuning, Tuning::Standard);
    ASSERT_EQ(replayDetector->numBlocks(), capacityBlocks);

    for (auto i = 0; i < capacityBlocks; ++i) {
        EXPECT_EQ(replayDetector->process(nullptr), liveEstimates[i]) << "block " << i;
    }
    EXPECT_EQ(replayDetector->numBlocksLeft(), 0);
    EXPECT_EQ(replayDetector->process(nullptr), 0.f);
}

TEST(ReplayPitchDetector, fromFileReturnsNullOnInvalidInput) {
    EXPECT_EQ(ReplayPitchDetector::fromFile(testUtils::getOutDir() / "doesNotExist.wav"), nullptr);
}
}  // namespace saint
