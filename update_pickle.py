#!/usr/bin/env python3
import pickle
import argparse
import os
import sys

def update_infos_file(pkl_path, old_prefix, new_prefix, backup=True):
    """
    pickle 파일 내부의 'lidar_path', 'annos_path', 'cuboids_path' 등을 찾아
    old_prefix → new_prefix 로 교체합니다.
    """
    print(f"\n--- Processing {pkl_path} ---")
    if not os.path.isfile(pkl_path):
        print(f"Error: '{pkl_path}' not found, skipping.")
        return

    # 1) 로드
    try:
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)
    except Exception as e:
        print(f"  [!] Failed to load pickle: {e}")
        return

    # 2) 수정
    update_count = 0
    def try_update(info, key):
        nonlocal update_count
        if key in info and isinstance(info[key], str):
            v = info[key]
            if v.startswith(old_prefix):
                info[key] = v.replace(old_prefix, new_prefix, 1)
                update_count += 1

    # 지원할 키 목록
    keys_to_fix = ['lidar_path', 'annos_path', 'cuboids_path']

    if isinstance(infos, list):
        for info in infos:
            if isinstance(info, dict):
                for key in keys_to_fix:
                    try_update(info, key)

    elif isinstance(infos, dict):
        for _, info in infos.items():
            if isinstance(info, dict):
                for key in keys_to_fix:
                    try_update(info, key)

    else:
        print("  [!] Unexpected pickle format, must be list or dict.")
        return

    print(f"  Updated {update_count} path entries.")

    # 3) 백업
    if backup:
        bak = pkl_path + '.backup'
        try:
            with open(bak, 'wb') as f:
                pickle.dump(infos, f)
            print(f"  Backup saved: {bak}")
        except Exception as e:
            print(f"  [!] Backup failed: {e}")
            return

    # 4) 덮어쓰기
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(infos, f)
        print("  [✔] File updated successfully")
    except Exception as e:
        print(f"  [!] Failed to write updated pickle: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Fix DATA_PATH prefixes inside Pandaset info pickle files"
    )
    parser.add_argument(
        'pkl_files', nargs='+',
        help="One or more .pkl files to fix"
    )
    parser.add_argument(
        '--old_prefix',
    # 이전 경로
        default='/media/cvlab/b641fc66-fd92-47de-84fa-c4c04f775df9/cvlab/software/OpenPCDet/data/pandaset',
        help="Old path prefix to replace"
    )
    parser.add_argument(
        '--new_prefix',
    # 지금 경로
        default='/home/q/dataset/pandaset',
        help="New (correct) path prefix"
    )
    parser.add_argument(
        '--no-backup', action='store_true',
        help="Skip creating .backup files"
    )
    args = parser.parse_args()

    for pkl in args.pkl_files:
        update_infos_file(
            pkl_path=pkl,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            backup=not args.no_backup
        )

if __name__ == '__main__':
    main()