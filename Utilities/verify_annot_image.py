# Compare two directory and print out all different files

from pathlib import Path
  
root_path = "/home/jsteeg/ufo_data/yolo_no_crop_vc/train"
folder_a = "annotations"
folder_b = "images"

files_a = set([x.stem for x in Path(root_path, folder_a).rglob("*.*") if x.is_file()])
files_b = set([x.stem for x in Path(root_path, folder_b).rglob("*.*") if x.is_file()])

print(f"{folder_a}: {len(files_a)}")
print(f"{folder_b}: {len(files_b)}")

diff_a = files_a - files_b
for f in diff_a:
    print(f)

diff_b = files_b - files_a
for f in diff_b:
    print(f)

print(f"Difference of {len(diff_a) + len(diff_b)} files")