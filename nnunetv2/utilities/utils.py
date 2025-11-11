#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from enum import Enum
import os
from pathlib import Path
import shutil
import threading
import os.path
from functools import lru_cache
from time import sleep, time
from typing import Union

import torch

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnunetv2.paths import nnUNet_raw
from multiprocessing import Pool


_CHMOD_LOCK = threading.Lock()


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files have a 4 digit channel index (_XXXX)
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def _should_copy(src: str, dst: str, follow_symlinks: bool) -> bool:
    """ Return True if dst must be (re)written. """
    if not os.path.exists(dst):
        return True
    if follow_symlinks:
        src_stat, dst_stat = os.stat(src), os.stat(dst)
    else:
        src_stat, dst_stat = os.lstat(src), os.lstat(dst)
    return src_stat.st_mtime_ns > dst_stat.st_mtime_ns


def copy_no_perms(src: str | os.PathLike[str], dst: str | os.PathLike[str], *, follow_symlinks: bool = True,
                  update: bool = False) -> str:
    """Copy without enforcing permissions change. Safe for some remote drive situation."""
    src, dst = os.fspath(src), os.fspath(dst)
    if update and not _should_copy(src, dst, follow_symlinks):
        return dst
    shutil.copyfile(src, dst, follow_symlinks=follow_symlinks)
    try:
        st = os.stat(src, follow_symlinks=follow_symlinks)
    except OSError:
        return dst
    try:
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=False)
    except OSError:
        pass
    with _CHMOD_LOCK:
        try:
            os.chown(dst, st.st_uid, st.st_gid, follow_symlinks=False)
        except (OSError, PermissionError, AttributeError):
            pass
    return dst


def create_paths_fn(folder, files, file_ending, f):
    p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))            
    return [join(folder, i) for i in files if p.fullmatch(i)]


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None, num_processes: int = 12) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []

    params_list = [(folder, files, file_ending, f) for f in identifiers]
    with Pool(processes=num_processes) as pool:
        list_of_lists = pool.starmap(create_paths_fn, params_list)

    return list_of_lists


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            expanded_label_file = os.path.expandvars(dataset[k]['label'])
            dataset[k]['label'] = os.path.abspath(join(raw_dataset_folder, expanded_label_file)) if not os.path.isabs(expanded_label_file) else expanded_label_file
            dataset[k]['images'] = [os.path.abspath(join(raw_dataset_folder, os.path.expandvars(i))) if not os.path.isabs(os.path.expandvars(i)) else os.path.expandvars(i) for i in dataset[k]['images']]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'], identifiers)
        segs = [join(raw_dataset_folder, 'labelsTr', i + dataset_json['file_ending']) for i in identifiers]
        dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    return dataset


class WaitFile:
    file: Path
    file_time: float
    max_wait: float
    WAIT_TIME: float = 0.2

    def __init__(self, file: Path, max_wait: float = 1E3) -> None:
        self.file = file
        if not file.exists():
            file.touch()
        self.max_wait = max_wait
        self.file_time = file.stat().st_mtime

    def reset(self):
        self.file_time = self.file.stat().st_mtime

    def wait(self) -> float:
        time_limit = time() + self.max_wait
        while True:
            if (new_time := self.file.stat().st_mtime) > self.file_time:
                self.file_time = new_time
                return new_time
            if (new_time := time()) > time_limit:
                raise TimeoutError(f'[error] fl client timeout waiting for {self.file}')
            sleep(self.WAIT_TIME)

    def read(self) -> dict[str, torch.Tensor]:
        return torch.load(self.file)

    def write(self, weights: dict[str, torch.Tensor]):
        return torch.save(weights, self.file)


class FileStatus(Enum):
    normal = 0
    reading = 1
    writing = 2


class Sentinel:
    WAIT_TIME: float = 0.2
    def __init__(self, file: Path, max_wait: float = 1000) -> None:
        self.sen_file: Path = file.with_name(file.stem + '-sen.txt')
        self.max_wait: float = max_wait
        if not self.sen_file.exists():
            _ = self.sen_file.write_text(str(FileStatus.normal.value))

    def status(self) -> FileStatus:
        return FileStatus(int(self.sen_file.read_text()))

    def update(self, status: FileStatus):
        _ = self.sen_file.write_text(str(status.value))

    def unlock(self):
        self.update(FileStatus.normal)

    def wait_read(self):
        time_limit = time() + self.max_wait
        while True:
            status = FileStatus(int(self.sen_file.read_text()))
            if status != FileStatus.writing:
                self.update(FileStatus.reading)
                return
            if time() > time_limit:
                raise TimeoutError(f'[error] fl client timeout waiting to read on sentinel {self.sen_file}')
            sleep(self.WAIT_TIME)

    def wait_write(self):
        time_limit = time() + self.max_wait
        while True:
            status = FileStatus(int(self.sen_file.read_text()))
            if status == FileStatus.normal:
                self.update(FileStatus.writing)
                return
            if time() > time_limit:
                raise TimeoutError(f'[error] fl client timeout waiting to read on sentinel {self.sen_file}')
            sleep(self.WAIT_TIME)


class DiffReader:
    sentinel: Sentinel
    file: WaitFile

    def __init__(self, file: Path, max_wait: float = 1E3) -> None:
        self.file = WaitFile(file, max_wait)
        self.sentinel = Sentinel(file)

    def read(self) -> dict[str, torch.Tensor]:
        _ = self.file.wait()
        self.sentinel.wait_read()
        res = self.file.read()
        self.sentinel.unlock()
        return res


class DiffWriter(DiffReader):
    def write(self, state_dict: dict[str, torch.Tensor]):
        self.sentinel.wait_write()
        self.file.write(state_dict)
        self.sentinel.unlock()


if __name__ == '__main__':
    print(get_filenames_of_train_images_and_targets(join(nnUNet_raw, 'Dataset002_Heart')))
