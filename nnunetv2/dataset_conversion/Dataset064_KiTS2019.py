import os
from pathlib import Path
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def convert_kits(kits_base_dir: str, task_name: str, nnunet_dataset_id: int):
    foldername = f"Dataset{nnunet_dataset_id:03d}_{task_name}"

    # setting up nnU-Net folders
    out_base = Path(os.environ.get('nnUNet_raw')).absolute().expanduser().joinpath(foldername)
    print(f"Copying data to {out_base}")
    imagestr = out_base.joinpath("imagesTr")
    imagests = out_base.joinpath("imagesTs")
    labelstr = out_base.joinpath("labelsTr")
    imagestr.mkdir(parents=True, exist_ok=True)
    imagests.mkdir(parents=True, exist_ok=True)
    labelstr.mkdir(parents=True, exist_ok=True)

    kits_base_path = Path(kits_base_dir).absolute().expanduser()

    idx = 0
    for tr in kits_base_path.glob('case_*'):
        tr = tr.name
        label = kits_base_path.joinpath(tr, 'segmentation.nii.gz')
        if label.exists():
            idx += 1
            shutil.copy(kits_base_path.joinpath(tr, 'imaging.nii.gz'), imagestr.joinpath(f'{tr}_0000.nii.gz'))
            shutil.copy(kits_base_path.joinpath(tr, 'segmentation.nii.gz'), labelstr.joinpath(f'{tr}.nii.gz'))
        else:
            shutil.copy(kits_base_path.joinpath(tr, 'imaging.nii.gz'), imagests.joinpath(f'{tr}_0000.nii.gz'))

    generate_dataset_json(str(out_base), {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
                          regions_class_order=(1, 3, 2),
                          num_training_cases=idx, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='prerelease',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="task_name")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted KiTS dataset (must have case_XXXXX subfolders)")
    parser.add_argument('task_name', type=str,
                        help="The task name to use")
    parser.add_argument('d', type=int, help='nnU-Net Dataset ID')
    args = parser.parse_args()
    convert_kits(args.input_folder, args.task_name, args.d)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset

