// Intentionally empty.
//
// pYIN's MonoPitchHMM.{h,cpp} carry `#include <boost/math/distributions.hpp>`,
// but no Boost symbol is actually used: the YIN threshold-distribution priors are
// precomputed static tables in YinUtil.cpp (uniformDist / betaDist1..4), not
// evaluated through Boost at runtime. This stub satisfies the include so the
// vendored pYIN core compiles without adding a Boost dependency to the project.
//
// If a future pYIN revision starts using Boost.Math for real, replace this stub
// with a genuine Boost dependency (header-only) in the Pyin CMakeLists.
#pragma once
