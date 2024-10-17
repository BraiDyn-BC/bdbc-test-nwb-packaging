from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import bdbc_session_explorer as sessx
import bdbc_nwb_packager as npack

load_dotenv()

DEBUG = bool(os.getenv('DEBUG'))

TASK_SESSION_INFO = json.loads(os.getenv('TASK_SESSION_INFO'))
TASK_SESSION_AVAILABILITY = json.loads(os.getenv('TASK_SESSION_AVAILABILITY'))
TASK_RAWDATA_PATH = Path(os.getenv('TASK_RAWDATA_PATH'))
TASK_BODY_VIDEO = Path(os.getenv('TASK_BODY_VIDEO_PATH'))
TASK_FACE_VIDEO = Path(os.getenv('TASK_FACE_VIDEO_PATH'))
TASK_EYE_VIDEO = Path(os.getenv('TASK_EYE_VIDEO_PATH'))


def load_paths() -> npack.paths.PathSettings:
    avail = sessx.session.Availability(**TASK_SESSION_AVAILABILITY)
    videos = npack.paths.VideoDataFiles(
        body=npack.paths.VideoDataFile(TASK_BODY_VIDEO, 0, 0, 0),  # FIXME: no metadata for the time being  # noqa: E501
        face=npack.paths.VideoDataFile(TASK_FACE_VIDEO, 0, 0, 0),  # FIXME: no metadata for the time being  # noqa: E501
        eye=npack.paths.VideoDataFile(TASK_EYE_VIDEO, 0, 0, 0)  # FIXME: no metadata for the time being  # noqa: E501
    )
    source = npack.paths.SourcePaths(
        rawdata=TASK_RAWDATA_PATH,
        videos=videos,
        deeplabcut=None,
        mesoscaler=None,
        pupilfitting=None
    )  # FIXME
    return npack.paths.PathSettings(
        session=sessx.Session(availability=avail, **TASK_SESSION_INFO),
        source=source,
        dlc_configs=None,
        destination=None,
    )


def test_paths():
    paths = load_paths()
    assert paths.session is not None
    assert paths.source.rawdata is not None
    assert paths.source.rawdata.exists()


def test_task_rawdata_handling():
    paths = load_paths()

    # metadata
    metadata = npack.metadata.metadata_from_rawdata(
        paths.session,
        paths.source.rawdata,
    )
    assert metadata is not None

    # timebase
    triggers, timebases = npack.packaging.core.load_timebases(
        metadata=metadata, rawfile=paths.source.rawdata, verbose=False
    )
    assert triggers is not None
    assert timebases is not None
    # assert triggers.B.size == timebases.B.size
    # assert triggers.V.size == timebases.V.size
    assert triggers.B.size <= timebases.B.size  # TODO: check if it is OK
    assert triggers.V.size <= timebases.V.size  # TODO: check if it is OK
    assert triggers.videos.size == timebases.videos.size
    assert_pulse_freq(timebases.B, 29, 31)
    assert_pulse_freq(timebases.V, 29, 31)
    assert_pulse_freq(timebases.videos, 99, 101)

    # trials
    trials = npack.packaging.trials.load_downsampled_trials(
        rawfile=paths.source.rawdata,
    )
    assert isinstance(trials, pd.DataFrame)
    assert trials.shape[1] < 10
    assert trials.shape[0] > 50


def assert_pulse_freq(timebase, fmin, fmax):
    assert timebase.size > 2
    fs = 1 / np.diff(timebase).mean()
    assert (fs >= fmin) and (fs <= fmax)
