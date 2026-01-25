## Benchmarking

### Metrics

Two metrics. CI could assess that neither of these has regressed.

#### Pitch detected yes/no

Accuracy of the algorithm in that respect can be evaluated objectively with a ROC score. The Area Under the Curve would be the first quality metric.

#### Accuracy of estimate

Easiest would be to have each file contain only one accurately tuned note. The ground truth for that file could then be a note number (e.g. E2). The RMS of the error could be the score assigned to each file. The average of all RMSs would then be the second quality metric.

### Test samples

At least 100 files. If CI is run on this repo, they will have to be redistributable. In https://github.com/saintmatthieu/loop-tempo-estimator, a script downloads a selection of files from https://freesound.org. Private forks of this repo would not have this constraint.

#### What each file must be

Each must be the recording of a single note, plucked once, so that the metrics described above apply.

While the user typically will adjust the pitch while using the tuner, the modulation in pitch that this causes is typically slow and is not believed to confuse the algorithm. On the other hand, accounting for this in the benchmarking would add lots of complexity. Hence we just rely on manual QA testing for this aspect.

#### Representativeness

The population of samples must be reasonably representative of the use cases. E.g. if 5% of the users are Ukulele players, then 5% of our test samples should be Ukulele samples.

Things to think of (please let me know or edit if I missed something):

- instrument
- string
- noise conditions
- microphone

### CI

The benchmarking can take the form of a "unit" test, with a pass or fail.

It will need access to the test bed whether run locally or by CI.

The test is light weight and can be run on each commit.

A red test should probably prevent merge.
