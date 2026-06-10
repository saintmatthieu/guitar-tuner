#pragma once

#ifdef __APPLE__
#include "CoreAudioInput.h"
namespace saint {
using AudioInput = CoreAudioInput;
}
#else
#include "AlsaAudioInput.h"
namespace saint {
using AudioInput = AlsaAudioInput;
}
#endif
