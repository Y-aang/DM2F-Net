import os
import shutil

def copy_and_rename_jpg_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        if os.path.isfile(src_file) and filename.lower().endswith('.jpg'):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_hazy{ext}"
            dst_file = os.path.join(dst_dir, new_filename)
            shutil.copy(src_file, dst_file)
            print(f"Copied and renamed {src_file} to {dst_file}")

# 设置源文件夹和目标文件夹路径
source_directory = '/root/workplace/dataset/HazeRD/data/simu'
destination_directory = '/root/workplace/dataset/HazeRD_formal/train/hazy'

# 调用函数
copy_and_rename_jpg_files(source_directory, destination_directory)
