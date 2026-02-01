#include <gtest/gtest.h>

#include <cmath>

#include "Utils.h"

namespace saint {
namespace {

TEST(QuadFit, QuadFit) {
    const float y1[] = {1.f, 2.f, 1.f};
    const auto r1 = utils::quadFit(y1);
    EXPECT_FLOAT_EQ(r1, 0.f);

    const float y2[] = {2.f, 3.f, 1.f};
    const auto r2 = utils::quadFit(y2);
    EXPECT_LT(r2, 0.f);

    const float y3[] = {1.f, 3.f, 2.f};
    const auto r3 = utils::quadFit(y3);
    EXPECT_GT(r3, 0.f);

    EXPECT_FLOAT_EQ(r2, -r3);
}
}  // namespace
}  // namespace saint
