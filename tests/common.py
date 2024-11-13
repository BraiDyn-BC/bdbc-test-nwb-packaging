# MIT License
#
# Copyright (c) 2024 Keisuke Sehara and Ryo Aoki
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

from pathlib import Path
from datetime import datetime
import os

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
        ('bodyvideo', 'TASK_HAS_BODY_VIDEO'),
        ('facevideo', 'TASK_HAS_FACE_VIDEO'),
        ('eyevideo', 'TASK_HAS_EYE_VIDEO'),
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


def setup_source_videos() -> npack.configure.SourceVideoFiles:
    videos_dir = os.getenv('TASK_VIDEOS_DIR')
    if (videos_dir is None) or (len(videos_dir.strip()) == 0):
        return npack.configure.SourceVideoFiles.empty()
    else:
        metadata = dict()
        for view, vtype in sessx.env.video_views().items():
            metadata[view] = sessx.videos.find_video_file(
                videos_dir,
                videotype=vtype
            )
        return npack.configure.SourceVideoFiles.from_paths(**metadata)


def load_paths() -> npack.configure.PathSettings:
    session = setup_session()
    videos  = sessx.video_files_from_session(
        session,
        videoroot=os.getenv('TASK_VIDEOS_ROOT'),
        error_handling='error',
    )
    dlcfiles = sessx.dlc_output_files_from_session(
        session,
        body=os.getenv('TASK_BODY_RESULTS_ROOT'),
        face=os.getenv('TASK_FACE_RESULTS_ROOT'),
        eye=os.getenv('TASK_EYE_RESULTS_ROOT'),
    )
    pupilfile = sessx.locate_pupil_file(
        session,
        pupilroot=os.getenv('TASK_EYE_RESULTS_ROOT'),
        locate_without_eyevideo=True,
    )
    mesofile = sessx.locate_mesoscaler_file(
        session,
        mesoroot=os.getenv('TASK_MESOSCALER_ROOT'),
    )
    source = npack.configure.SourcePaths(
        rawdata=TASK_RAWDATA_PATH,
        videos=videos,
        deeplabcut=dlcfiles,
        mesoscaler=mesofile,
        pupilfitting=pupilfile,
    )
    dest = npack.configure.setup_destination_paths(
        session=session,
        nwbroot=TEMP_NWBROOT,
    )
    return npack.configure.PathSettings(
        session=session,
        source=source,
        dlc_configs=None,
        destination=dest,
    )
