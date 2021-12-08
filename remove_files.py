import os
import shutil
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    data_dirs = args.data_dir
    videos = [x for x in natsorted(os.listdir(data_dirs)) if os.path.isdir(os.path.join(data_dirs, x))]
    invalid_videos = [os.path.join(data_dirs, video) for video in videos if 'failed.txt' in os.listdir(os.path.join(data_dirs, video))]
    for v in invalid_videos:
        shutil.rmtree(v)
    