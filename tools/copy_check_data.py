import os
import glob
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_scene_pseudo_depth(folder, root_directory, target_directory):
    try:
        scene_id = int(folder)
        folder_path = os.path.join(root_directory, folder)
        if not os.path.isdir(folder_path):
            return False

        example_path = os.path.join(root_directory, folder, "images")
        gt_depth_path = os.path.join(root_directory, folder, "depth_flows_4")

        images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))
        gt_depths = sorted(glob.glob(os.path.join(gt_depth_path, "*.npy")))
        pseudo_path = os.path.join(root_directory, folder, "pseudo_depth_4")

        pseudo_depths_confs = sorted(glob.glob(os.path.join(pseudo_path, "*.npy")))
        if len(pseudo_depths_confs) != 2 * len(images):
            print(f"Nums of pseudo_depths_confs: {len(pseudo_depths_confs)}, Nums of images: {len(images)}")
            print(f"Scene: {scene_id} has not been processed")
            return False

        target_pseudo_path = os.path.join(target_directory, folder, "pseudo_depth_4")
        os.makedirs(os.path.dirname(target_pseudo_path), exist_ok=True)

        source_files = sorted(glob.glob(os.path.join(pseudo_path, "*.npy")))
        target_files = sorted(glob.glob(os.path.join(target_pseudo_path, "*.npy")))

        if len(source_files) == len(target_files):
            return True  # Already copied

        # Execute cp -r pseudo_path/. target_pseudo_path
        src_with_dot = os.path.join(pseudo_path, ".")
        result = subprocess.run(
            ["cp", "-r", src_with_dot, target_pseudo_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Verification
        target_files_after = sorted(glob.glob(os.path.join(target_pseudo_path, "*.npy")))
        if len(source_files) != len(target_files_after):
            print(f"❌ Copy failed for scene {scene_id}: file count mismatch "
                  f"({len(source_files)} vs {len(target_files_after)})")
            return False

        return True

    except Exception as e:
        print(f"❌ Error processing scene {folder}: {e}")
        return False

def verify_all_scenes(root_directory, target_directory):
    print("\n🔍 Starting final verification of all scenes...")
    folders = sorted(os.listdir(root_directory))
    mismatched = []

    for folder in tqdm(folders, desc="Verifying"):
        try:
            scene_id = int(folder)
        except ValueError:
            continue  # skip non-numeric folders

        src_pseudo = os.path.join(root_directory, folder, "pseudo_depth_4")
        tgt_pseudo = os.path.join(target_directory, folder, "pseudo_depth_4")

        if not os.path.isdir(src_pseudo):
            print(f"⚠️  Source pseudo_depth_4 missing for scene {folder}")
            continue

        src_files = sorted(glob.glob(os.path.join(src_pseudo, "*.npy")))
        tgt_files = sorted(glob.glob(os.path.join(tgt_pseudo, "*.npy")))

        if len(src_files) != len(tgt_files):
            mismatched.append((folder, len(src_files), len(tgt_files)))

    if mismatched:
        print(f"\n❌ Found {len(mismatched)} scenes with mismatched file counts:")
        for scene, src_cnt, tgt_cnt in mismatched:
            print(f"  Scene {scene}: source={src_cnt}, target={tgt_cnt}")
    else:
        print("✅ All scenes verified successfully. File counts match.")

    return len(mismatched) == 0

def verify_no_empty_npy(target_directory):
    print("\n🔍 Checking for 0-byte .npy files in target pseudo_depth_4 folders...")
    empty_files = []

    # Traverse all scene directories
    for folder in tqdm(os.listdir(target_directory), desc="Scanning scenes"):
        if not folder.isdigit():
            continue
        pseudo_dir = os.path.join(target_directory, folder, "pseudo_depth_4")
        if not os.path.isdir(pseudo_dir):
            continue

        try:
            with os.scandir(pseudo_dir) as entries:
                for entry in entries:
                    if entry.name.endswith('.npy') and entry.is_file():
                        if entry.stat().st_size == 0:
                            empty_files.append(os.path.join(pseudo_dir, entry.name))
        except OSError as e:
            print(f"⚠️  Cannot scan {pseudo_dir}: {e}")

    if empty_files:
        print(f"\n❌ Found {len(empty_files)} empty (0-byte) .npy files:")
        for f in empty_files[:10]:  # Only print first 10 to avoid flooding
            print(f"  {f}")
        if len(empty_files) > 10:
            print(f"  ... and {len(empty_files) - 10} more.")
    else:
        print("✅ No empty .npy files found. All look good!")

    return len(empty_files) == 0

# Main process
def main(root_directory, target_directory, max_workers=8):
    folders = sorted(os.listdir(root_directory))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_folder = {
            executor.submit(copy_scene_pseudo_depth, folder, root_directory, target_directory): folder
            for folder in folders
        }

        # Display progress with tqdm
        for future in tqdm(as_completed(future_to_folder), total=len(folders), desc="Copying scenes"):
            folder = future_to_folder[future]
            try:
                success = future.result()
                if not success:
                    pass  # Error already printed in function
            except Exception as e:
                print(f"Unexpected error for folder {folder}: {e}")

     # Final alignment check
    verify_all_scenes(root_directory, target_directory)

    # Check for 0KB .npy files in target directory only
    verify_no_empty_npy(target_directory)

# Example usage
if __name__ == "__main__":
    root_directory = "xxx/SLARM_data/datasets/waymo/training"
    target_directory = "xxx/SLARM_data/datasets/waymo/training"
    main(root_directory, target_directory, max_workers=8)
