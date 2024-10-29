from pathlib import Path
from datetime import datetime
import os
import shutil

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import bdbc_session_explorer as sessx
import bdbc_nwb_packager as npack

load_dotenv()

DEBUG = bool(os.getenv('DEBUG'))

TEMP_NWBROOT = Path(os.getenv('TEMP_NWBROOT_DIR'))

TASK_RAWDATA_PATH = Path(os.getenv('TASK_RAWDATA_PATH'))


def setup_session_availability() -> sessx.session.Availability:
    avail = dict()
    for name, env in (
        ('rawdata', 'TASK_RAWDATA_PATH'),
        ('bodyvideo', 'TASK_BODY_VIDEO_PATH'),
        ('facevideo', 'TASK_FACE_VIDEO_PATH'),
        ('eyevideo', 'TASK_EYE_VIDEO_PATH'),
    ):
        val = os.getenv(env)
        avail[name] = ((val is not None) and len(val.strip()) > 0)
    return sessx.session.Availability(**avail)


def setup_session(session_type='task') -> sessx.session.Session:
    def _as_date(datestr) -> datetime:
        return datetime.strptime(datestr, '%y%m%d')
    TYPE = session_type.upper()
    metadata = dict()
    for key, env, typ in (
        ('batch', 'BATCH', str),
        ('animal', 'ANIMAL', str),
        ('date', 'DATE', _as_date),
        ('type', 'TYPE', str),
        ('dayindex', 'DAYIDX', int),
        ('sessionindex', 'SESSIDX', int),
        ('description', 'DESC', str),
        ('comments', 'COMMENTS', str),
    ):
        metadata[key] = typ(os.getenv(f'{TYPE}_SESSION_{env}'))
    return sessx.session.Session(
        availability=setup_session_availability(),
        **metadata
    )


def setup_source_videos() -> npack.paths.VideoDataFiles:
    metadata = dict()
    videos_dir = os.getenv('TASK_VIDEOS_DIR')
    if (videos_dir is None) or (len(videos_dir.strip()) == 0):
        for key in ('body', 'face', 'eye'):
            metadata[key] = npack.paths.VideoDataFile.empty()
    else:
        for key, label in sessx.videos.VideoFiles.LABELS.items():
            videofile = sessx.videos.find_video_file(
                videos_dir,
                videotype=label
            )
            metadata[key] = npack.paths.setup_video_file(videofile)
    return npack.paths.VideoDataFiles(**metadata)


def load_paths() -> npack.paths.PathSettings:
    session = setup_session()
    videos = setup_source_videos()
    source = npack.paths.SourcePaths(
        rawdata=TASK_RAWDATA_PATH,
        videos=videos,
        deeplabcut=None,
        mesoscaler=None,
        pupilfitting=None
    )  # FIXME
    dest = npack.paths.setup_destination_paths(
        session=session,
        nwbroot=TEMP_NWBROOT,
    )
    return npack.paths.PathSettings(
        session=session,
        source=source,
        dlc_configs=None,
        destination=dest,
    )


def test_paths():
    paths = load_paths()
    assert paths.session is not None
    assert paths.source.rawdata is not None
    assert paths.source.rawdata.exists()


def test_base():
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
    assert triggers.B.size <= timebases.B.size  # TODO: check if it is OK
    assert triggers.V.size <= timebases.V.size  # TODO: check if it is OK
    assert triggers.videos.size == timebases.videos.size
    assert_pulse_freq(timebases.B, 29, 31)
    assert_pulse_freq(timebases.V, 29, 31)
    assert_pulse_freq(timebases.videos, 99, 101)


def test_trials():
    paths = load_paths()

    # metadata
    metadata = npack.metadata.metadata_from_rawdata(
        paths.session,
        paths.source.rawdata,
    )

    # timebase
    triggers, timebases = npack.packaging.core.load_timebases(
        metadata=metadata, rawfile=paths.source.rawdata, verbose=False
    )

    # trials
    trials = npack.packaging.trials.load_trials(
        rawfile=paths.source.rawdata,
    )
    assert isinstance(trials, pd.DataFrame)
    assert trials.shape[1] < 10
    assert trials.shape[0] > 50

    # downsampled trials
    trials_ds = npack.packaging.trials.load_downsampled_trials(
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


def test_rawdata_nwb():
    """tests writing the DAQ-log entries and trials to the NWBFile object"""
    if TEMP_NWBROOT.exists():
        shutil.rmtree(TEMP_NWBROOT)
    paths = load_paths()
    nwbfile = npack.packaging.package_nwb(
        paths,
        copy_videos=False,
        register_rois=False,
        write_imaging_frames=False
    )
    assert (nwbfile is not None)
