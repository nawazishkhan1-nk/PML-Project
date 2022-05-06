import SimpleITK as sitk
import glob
import os
import shapeworks as sw


DATA_PATH = '/home/sci/nawazish.khan/PML-Project-data/data/training/label'
PROJECT_PATH = '/home/sci/nawazish.khan/PML-Project/'
data_dir = f"{PROJECT_PATH}/data"
def convert_nifti():
    data_dir = f"{PROJECT_PATH}/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    input_files = glob.glob(f"{DATA_PATH}/*.mhd")
    print(f"Loading {len(input_files)} CT Samples from Training Data Set")

    for input_file in input_files:
        file_name = input_file.split('/')[-1].split('.')[0]
        print(f"Converting {file_name} .....")
        img = sitk.ReadImage(input_file)
        sitk.WriteImage(img, f"{data_dir}/{file_name}.nii")
        
def compress_segmentations():
    seg_files = glob.glob(f"{data_dir}/*.nii")
    print(f"Loading {len(seg_files)} Segmentations")
    for seg_file in seg_files:
        file_name = seg_file.split('/')[-1].split('.')[0]
        print(f"Compressing {file_name} .....")
        img = sw.Image(seg_file).write(f"{data_dir}/{file_name}.nrrd", compressed=True)

convert_nifti()
compress_segmentations()
