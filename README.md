# Benchmarking

## Metrics

Two metrics. CI could assess that neither of these has regressed.

### Pitch detected yes/no

Accuracy of the algorithm in that respect can be evaluated objectively with a ROC score. The Area Under the Curve would be the first quality metric.

### Accuracy of estimate

Easiest would be to have each file contain only one accurately tuned note. The ground truth for that file could then be a note number (e.g. E2). The RMS of the error could be the score assigned to each file. The average of all RMSs would then be the second quality metric.

## Test recordings

### Guitar samples

In `./eval/testFiles/notes/`

#### What each file must be

- one note plucked once
- the note must be VERY well tuned
- don't mute the other strings
- background as clean as possible (noise is added artificially by the benchmarking code)

While the user typically will adjust the pitch while using the tuner, the modulation in pitch that this causes is typically slow and is not believed to confuse the algorithm. On the other hand, accounting for this in the benchmarking would add lots of complexity. Hence we just rely on manual QA testing for this aspect.

#### Representativeness

The population of samples must be reasonably representative of the use cases. E.g. if 5% of the users are Ukulele players, then 5% of our test samples should be Ukulele samples.

Things to think of (please let me know or edit if I missed something):

- instrument
- string
- noise conditions
- microphone

### Noise

In `./eval/testFiles/noise/`

Noise profiles are recorded separately and are mixed automatically by the benchmarking code before running the algorithm.

- Aim between 15 and 20 seconds (they are looped if the note recording is longer)
- no need to aim for a certain loudness - the benchmarking will calibrate loudness before mixing. Just make sure it doesn't clip

Types of noise:

- Air conditioner
- Fan of a computer
- Kitchen / cooking background noises
- Kids in the background
- TV / Radio behind doors
- ...

## CI

The benchmarking can take the form of a "unit" test, with a pass or fail.

It will need access to the test bed whether run locally or by CI.

The test is light weight and can be run on each commit.

A red test should probably prevent merge.
