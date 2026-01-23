#include "testUtils.h"

#include "sndfile.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace saint {
namespace fs = std::filesystem;

testUtils::Audio testUtils::fromWavFile(fs::path path) {
  // read all the file in one go using libsndfile:
  SF_INFO sfinfo;
  SNDFILE *sndfile = sf_open(path.string().c_str(), SFM_READ, &sfinfo);
  if (sndfile == nullptr) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  std::vector<float> audio(sfinfo.frames * sfinfo.channels);
  sf_count_t numRead = sf_readf_float(sndfile, audio.data(), sfinfo.frames);
  sf_close(sndfile);
  if (numRead != sfinfo.frames) {
    throw std::runtime_error("Could not read all samples from file: " +
                             path.string());
  }
  return {std::move(audio), sfinfo.samplerate};
}

fs::path testUtils::getEvalDir() {
  return fs::path(__FILE__).parent_path() / ".." / ".." / "eval";
}

fs::path testUtils::getOutDir() { return getEvalDir() / "out"; }
} // namespace saint
