#include "testUtils.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>

#include "sndfile.h"

namespace saint {
namespace fs = std::filesystem;

std::optional<testUtils::Audio> testUtils::fromWavFile(fs::path path, int numSamples) {
    // read all the file in one go using libsndfile:
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(path.string().c_str(), SFM_READ, &sfinfo);
    if (sndfile == nullptr) {
        return std::nullopt;
    }
    const auto framesToRead =
        numSamples > 0 ? std::min<sf_count_t>(numSamples, sfinfo.frames) : sfinfo.frames;

    std::vector<float> audio(framesToRead * sfinfo.channels);
    sf_count_t numRead = sf_readf_float(sndfile, audio.data(), framesToRead);
    sf_close(sndfile);
    if (numRead != framesToRead) {
        return std::nullopt;
    }

    const auto channelFormat = sfinfo.channels == 1 ? ChannelFormat::Mono : ChannelFormat::Stereo;
    return Audio{std::move(audio), sfinfo.samplerate, channelFormat};
}

bool testUtils::toWavFile(fs::path path, const Audio& audio, TeeStream* logger,
                          const std::string& what) {
    SF_INFO sfinfo;
    sfinfo.channels = audio.channelFormat == ChannelFormat::Mono ? 1 : 2;
    sfinfo.samplerate = audio.sampleRate;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    if (!std::filesystem::exists(path.parent_path())) {
        std::filesystem::create_directories(path.parent_path());
    }

    SNDFILE* sndfile = sf_open(path.string().c_str(), SFM_WRITE, &sfinfo);
    if (sndfile == nullptr) {
        std::cerr << "Could not open file for writing: " << path << "\n";
        return false;
    }
    const auto numFrames = audio.interleaved.size() / sfinfo.channels;
    sf_count_t numWritten = sf_writef_float(sndfile, audio.interleaved.data(), numFrames);
    sf_close(sndfile);
    const auto success = numWritten == static_cast<sf_count_t>(numFrames);
    if (!success) {
        std::cerr << "Could not write all samples to file: " << path << "\n";
    } else if (logger != nullptr) {
        if (!what.empty()) {
            *logger << what;
        } else {
            *logger << "Wrote";
        }
        *logger << "\t" << path << "\n";
    }
    return success;
}

fs::path testUtils::getEvalDir() {
    return fs::path(__FILE__).parent_path() / ".." / ".." / "eval";
}

fs::path testUtils::getOutDir() {
    const auto dir = getEvalDir() / "out";
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    return dir;
}

void testUtils::scaleToRms(std::vector<float>& data, float targetRmsDb) {
    float sumSquares = 0.f;
    float peak = 0.f;
    for (const auto sample : data) {
        sumSquares += sample * sample;
        peak = std::max(peak, std::abs(sample));
    }
    const float currentRms = std::sqrt(sumSquares / static_cast<float>(data.size()));
    const float targetRms = std::pow(10.f, targetRmsDb / 20.f);

    constexpr auto maxPeakDb = -30.f;
    const float maxPeak = std::pow(10.f, maxPeakDb / 20.f);

    const float rmsScale = targetRms / currentRms;
    const float peakScale = maxPeak / peak;
    const float scale = std::min(rmsScale, peakScale);
    for (auto& sample : data) {
        sample *= scale;
    }
}

void testUtils::mixNoise(Audio& signal, const std::vector<float>& noise) {
    const auto noiseSize = noise.size();
    size_t n = 0;
    const auto numChannels = signal.channelFormat == ChannelFormat::Mono ? 1 : 2;
    for (auto i = 0; i < signal.interleaved.size() / numChannels; ++i) {
        for (auto c = 0; c < numChannels; ++c) {
            signal.interleaved[i * numChannels + c] += noise[n];
        }
        n = (n + 1) % noiseSize;
    }
}

float testUtils::midiNoteToFrequency(int midiNote) {
    return 440.f * std::pow(2.f, (midiNote - 69) / 12.f);
}

float testUtils::getTrueFrequency(const std::filesystem::path& filePath) {
    const auto filename = filePath.stem().string();
    // File name in the form of <note name><note octave>:
    const auto noteName = filename.substr(0, filename.size() - 1);
    const auto noteOctave = filename.back() - '0';
    const std::vector<std::string> noteNames{
        "C", "D", "E", "F", "G", "A", "B",
    };
    const std::vector<int> noteOffsets{
        0, 2, 4, 5, 7, 9, 11,
    };
    const auto it =
        std::find_if(noteNames.begin(), noteNames.end(), [&noteName](const std::string& name) {
            return std::equal(name.begin(), name.end(), noteName.begin(), noteName.end(),
                              [](char a, char b) { return std::tolower(a) == std::tolower(b); });
        });

    if (it == noteNames.end()) {
        return 0.f;
    }

    const int noteOffset = noteOffsets[std::distance(noteNames.begin(), it)];
    const int midiNote = (noteOctave + 1) * 12 + noteOffset;
    return midiNoteToFrequency(midiNote);
}

fs::path testUtils::getFileShortName(const fs::path& filePath) {
    return filePath.parent_path().stem() / filePath.stem();
}

std::optional<testUtils::Sample> testUtils::getSampleFromFile(const fs::path& filePath) {
    const auto trueFreq = getTrueFrequency(filePath);
    if (trueFreq == 0.f) {
        std::cerr << "Could not determine true frequency for " << filePath << "\n";
        return std::nullopt;
    }

    const auto labelPath = filePath.parent_path() / (filePath.stem().string() + ".txt");
    if (!fs::exists(labelPath)) {
        std::cerr << "Could not find label file: " << labelPath << "\n";
        return std::nullopt;
    }

    // Formatted in Audacity label format: <start time>tab<end time> (ignore
    // possible label text)
    std::ifstream labelFile(labelPath);
    std::string line;
    if (!std::getline(labelFile, line)) {
        std::cerr << "Could not read label file: " << labelPath << "\n";
        return std::nullopt;
    }

    const auto tabPos = line.find('\t');
    if (tabPos == std::string::npos) {
        std::cerr << "Could not parse label file: " << labelPath << "\n";
        return std::nullopt;
    }
    const auto startTime = std::stof(line.substr(0, tabPos));
    const auto endTime = std::stof(line.substr(tabPos + 1));

    return Sample{filePath, Truth{startTime, endTime, trueFreq}};
}

void testUtils::writeLogMarks(const fs::path& filenameStem, int sampleRate, Marking marking) {
    // Write a text file that can be imported by Audacity as labels:
    const auto labelPath = getOutDir() / "wav" / (filenameStem.string() + "_log_marks.txt");
    std::ofstream labelFile(labelPath);
    labelFile << static_cast<double>(marking.startSample) / sampleRate << "\t"
              << static_cast<double>(marking.endSample) / sampleRate << "\n";
}

double testUtils::writeResultFile(const Sample& sample, const std::vector<Result>& results,
                                  const fs::path& outputPath) {
    if (!fs::exists(outputPath.parent_path())) {
        fs::create_directories(outputPath.parent_path());
    }

    std::ofstream resultFile(outputPath);

    double rmsErrorCents = 0.;
    std::vector<double> errorCents;
    for (const auto& r : results) {
        if (r.f > 0.) {
            const auto e = 1200. * std::log2(r.f / sample.truth.frequency);
            rmsErrorCents += e * e;
            errorCents.push_back(e);
        }
    }
    if (!errorCents.empty()) {
        rmsErrorCents = std::sqrt(rmsErrorCents / errorCents.size());
    }
    resultFile << "rmsErrorCents = " << rmsErrorCents << "\n";

    resultFile << "results = [";
    auto separator = "";
    for (const auto& e : errorCents) {
        resultFile << separator << e;
        separator = ",\n";
    }
    resultFile << "]\n";

    return rmsErrorCents;
}

}  // namespace saint
