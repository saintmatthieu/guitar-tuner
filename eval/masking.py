import math

# peak: -21 at 89
# neighbour: -40 at 60 no

# peak: -21 at 89
# neighbour: -35 at 185 yes

# peak: -35 at 185
# neighbour: -62 at 137 no

# peak: -30 at 373
# neighbour: -44 at 459 yes

# peak: -60 at 1876
# neighbour: -68 at 1973 yes

# peak: -60 at 1876
# neighbour: -78 at 1842 no

# peak: -20 at 61
# neighbour: -44 at 22 yes

# peak: -44 at 341
# neighbour: -63 at 309 yes

binFreq = 2.69165039
def to_bark(bin):
    f = bin * binFreq
    return 13 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500) ** 2)

# Function that returns the Bark difference between two bins
def bark_difference(bin1, bin2):
    bark1 = to_bark(bin1)
    bark2 = to_bark(bin2)
    return abs(bark1 - bark2)


samples = [{"peak": {"dB": -21, "bin": 89}, "neighbour": {"dB": -40, "bin": 60, "good": False}},
            {"peak": {"dB": -21, "bin": 89}, "neighbour": {"dB": -35, "bin": 185, "good": True}},
            {"peak": {"dB": -35, "bin": 185}, "neighbour": {"dB": -62, "bin": 137, "good": False}},
            {"peak": {"dB": -30, "bin": 373}, "neighbour": {"dB": -44, "bin": 459, "good": True}},
            {"peak": {"dB": -60, "bin": 1876}, "neighbour": {"dB": -68, "bin": 1973, "good": True}},
            {"peak": {"dB": -60, "bin": 1876}, "neighbour": {"dB": -78, "bin": 1842, "good": False}},
            {"peak": {"dB": -20, "bin": 61}, "neighbour": {"dB": -44, "bin": 22, "good": True}},
            {"peak": {"dB": -44, "bin": 341}, "neighbour": {"dB": -63, "bin": 309, "good": True}}]
# for each sample, calculate (neighbour.dB - peak.dB) / log2(abs(neighour.bin - peak.bin)) and print the result, as well if it's a good neighbour or not.
for sample in samples:
    peak_dB = sample["peak"]["dB"]
    peak_bin = sample["peak"]["bin"]
    neighbour_dB = sample["neighbour"]["dB"]
    neighbour_bin = sample["neighbour"]["bin"]
    result = (neighbour_dB - peak_dB) / bark_difference(neighbour_bin, peak_bin)
    print(f"dB/Bark: {result}, Good: {sample['neighbour']['good']}")

# Now print the dB/Bark for the good and then for the bad neighbours:
print("\nGood neighbours:")
for sample in samples:
    if sample["neighbour"]["good"]:
        peak_dB = sample["peak"]["dB"]
        peak_bin = sample["peak"]["bin"]
        neighbour_dB = sample["neighbour"]["dB"]
        neighbour_bin = sample["neighbour"]["bin"]
        result = (neighbour_dB - peak_dB) / bark_difference(neighbour_bin, peak_bin)
        print(f"dB/Bark: {result}")
      

print("\nBad neighbours:")
for sample in samples:
    if not sample["neighbour"]["good"]:
        peak_dB = sample["peak"]["dB"]
        peak_bin = sample["peak"]["bin"]
        neighbour_dB = sample["neighbour"]["dB"]
        neighbour_bin = sample["neighbour"]["bin"]
        result = (neighbour_dB - peak_dB) / bark_difference(neighbour_bin, peak_bin)
        print(f"dB/Bark: {result}")