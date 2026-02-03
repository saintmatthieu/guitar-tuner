#include <gtest/gtest.h>

#include <cmath>

#include "Utils.h"

namespace saint {
namespace {

TEST(FastDb, FastDb) {
    const float power[] = {0.0001f, 0.001f, 0.01f, 0.1f, 1.f, 10.f, 100.f, 1000.f, 10000.f};
    for (const auto& v : power) {
        const auto expected = 10.f * std::log10(v);
        const auto actual = utils::FastDb(v);
        EXPECT_NEAR(expected, actual, 0.04f);
    }
}

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

TEST(GCD, GCD) {
    {
        const std::vector<float> values = {0.f};
        const auto gcd = utils::getApproximateGcd(values);
        EXPECT_FLOAT_EQ(gcd, 0.f);
    }
    {
        const std::vector<float> values = {5.f};
        const auto gcd = utils::getApproximateGcd(values);
        EXPECT_FLOAT_EQ(gcd, 0.f);
    }
    {
        const std::vector<float> values = {4.f, 6.f, 8.f};
        const auto gcd = utils::getApproximateGcd(values);
        EXPECT_EQ(gcd, 2.f);
    }
    {
        const std::vector<float> values = {5.1f, 10.2f, 15.3f};
        const auto gcd = utils::getApproximateGcd(values);
        EXPECT_EQ(gcd, 5.1f);
    }
}

TEST(doubleCheckEstimate, doubleCheckEstimate) {
    constexpr auto N = 5;
    std::vector<float> dbSpectrum(100, -100.f);
    dbSpectrum[20] = dbSpectrum[40] = 0.f;
    const auto result = utils::doubleCheckEstimate(20, dbSpectrum, 0, N);
    EXPECT_EQ(result, 20.f);
    dbSpectrum[30] = 0.f;
    const auto result2 = utils::doubleCheckEstimate(20, dbSpectrum, 0, N);
    EXPECT_EQ(result2, 10.f);
    dbSpectrum[25] = 0.f;
    const auto result3 = utils::doubleCheckEstimate(20, dbSpectrum, 0, N);
    EXPECT_EQ(result3, 5.f);
}
}  // namespace
}  // namespace saint
