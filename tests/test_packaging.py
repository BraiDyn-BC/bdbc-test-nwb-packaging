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

import os
import shutil

import bdbc_nwb_packager as npack

from common import (
    TEMP_NWBROOT,
    load_paths as _load_paths,
    setup_session as _setup_session,
)


def test_file_packaging():
    """tests writing the DAQ-log entries and trials to the NWBFile object"""
    if TEMP_NWBROOT.exists():
        shutil.rmtree(TEMP_NWBROOT)
    npack.logging.get_logger(file_output=True, prefix='test-nwb-packaging_')
    paths = _load_paths()
    nwbfile = npack.packaging.process(
        paths.session,
        copy_videos=False,
        register_rois=True,
        write_imaging_frames=False,
        rawroot=os.getenv('TASK_RAWDATA_ROOT'),
        mesoroot=os.getenv('TASK_MESOSCALER_ROOT'),
        videoroot=os.getenv('TASK_VIDEOS_ROOT'),
        body_results_root=os.getenv('TASK_BODY_RESULTS_ROOT'),
        face_results_root=os.getenv('TASK_FACE_RESULTS_ROOT'),
        eye_results_root=os.getenv('TASK_EYE_RESULTS_ROOT'),
        pupilroot=os.getenv('TASK_EYE_RESULTS_ROOT'),
        bodymodeldir=os.getenv('TASK_BODY_MODEL_DIR'),
        facemodeldir=os.getenv('TASK_FACE_MODEL_DIR'),
        eyemodeldir=os.getenv('TASK_EYE_MODEL_DIR'),
        nwbroot=os.getenv('TEMP_NWBROOT_DIR'),
    )
    assert (nwbfile is not None)


def __test_read_imaging_data():
    if TEMP_NWBROOT.exists():
        shutil.rmtree(TEMP_NWBROOT)
    paths = _load_paths()
    metadata = npack.file_metadata.read_recordings_metadata(
        paths.session,
        paths.source.rawdata,
    )
    triggers, timebases = npack.timebases.read_timebases(
        metadata=metadata, rawfile=paths.source.rawdata, verbose=False
    )
    frames = npack.imaging.load_imaging_data(
        rawfile=paths.source.rawdata,
        timebases=timebases,
    )
    assert frames.has_data()


def __test_full_packaging():
    if TEMP_NWBROOT.exists():
        shutil.rmtree(TEMP_NWBROOT)
    sess = _setup_session()
    npack.packaging.process(
        sess,
        rawroot=os.getenv('TASK_RAWDATA_ROOT'),
        mesoroot=os.getenv('TASK_MESOSCALER_ROOT'),
        videoroot=os.getenv('TASK_VIDEOS_ROOT'),
        body_results_root=os.getenv('TASK_BODY_RESULTS_ROOT'),
        face_results_root=os.getenv('TASK_FACE_RESULTS_ROOT'),
        eye_results_root=os.getenv('TASK_EYE_RESULTS_ROOT'),
        pupilroot=os.getenv('TASK_EYE_RESULTS_ROOT'),
        bodymodeldir=os.getenv('TASK_BODY_MODEL_DIR'),
        facemodeldir=os.getenv('TASK_FACE_MODEL_DIR'),
        eyemodeldir=os.getenv('TASK_EYE_MODEL_DIR'),
        nwbroot=os.getenv('TEMP_NWBROOT_DIR'),
    )
