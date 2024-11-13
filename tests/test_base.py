# MIT License
#
# Copyright (c) 2024 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd

import bdbc_nwb_packager as npack

from common import (
    load_paths as _load_paths,
)


def test_paths():
    paths = _load_paths()
    assert paths.session is not None
    assert paths.source.rawdata is not None
    assert paths.source.rawdata.exists()


def test_timebases():
    paths = _load_paths()

    # metadata
    metadata = npack.file_metadata.read_recordings_metadata(
        paths.session,
        paths.source.rawdata,
    )
    assert metadata is not None

    # timebase
    triggers, timebases = npack.timebases.read_timebases(
        metadata=metadata, rawfile=paths.source.rawdata, verbose=False
    )
    assert triggers is not None
    assert timebases is not None
    assert triggers.B.size <= timebases.B.size  # TODO: check if it is OK
    assert triggers.V.size <= timebases.V.size  # TODO: check if it is OK
    assert triggers.videos.size == timebases.videos.size
    assert_pulse_freq(timebases.B, 29, 31)
    assert_pulse_freq(timebases.V, 29, 31)
    assert_pulse_freq(timebases.videos, 99, 101)


def test_trials():
    paths = _load_paths()

    # metadata
    metadata = npack.file_metadata.read_recordings_metadata(
        paths.session,
        paths.source.rawdata,
    )

    # timebase
    triggers, timebases = npack.timebases.read_timebases(
        metadata=metadata, rawfile=paths.source.rawdata, verbose=False
    )

    # trials
    trials = npack.trials.load_trials(
        rawfile=paths.source.rawdata,
    )
    assert isinstance(trials, pd.DataFrame)
    assert trials.shape[1] < 10
    assert trials.shape[0] > 50

    # downsampled trials
    trials_ds = npack.trials.load_downsampled_trials(
        rawfile=paths.source.rawdata,
    )
    assert isinstance(trials_ds, pd.DataFrame)

    # compare raw vs downsampled
    assert trials_ds.shape == trials.shape
    for column in (
        'reaction_time',
        'trial_outcome',
        'pull_duration_for_success'
    ):
        assert np.all(nancompare(
            trials[column].values, trials_ds[column].values
        ))
    for column in (
        'pull_onset',
        'trial_start',
        'trial_end',
    ):
        assert np.all(nancompare(
            trials[column].values, trials_ds[column].values, tol=0.05,
        ))


def assert_pulse_freq(timebase, fmin, fmax):
    assert timebase.size > 2
    fs = 1 / np.diff(timebase).mean()
    assert (fs >= fmin) and (fs <= fmax)


def nancompare(vec1, vec2, tol=1e-6):
    valid1 = ~np.isnan(vec1)
    valid2 = ~np.isnan(vec2)
    valid = np.array(valid1 == valid2)
    valid_both = np.logical_and(valid1, valid2)
    valid[valid_both] = np.abs(vec1[valid_both] - vec2[valid_both]) <= tol
    return valid
