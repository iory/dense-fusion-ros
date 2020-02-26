#!/usr/bin/env python

import argparse
import os
import os.path as osp

import gdown
import rospkg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'dense_fusion_ros'

    ycb_mesh_models_path = osp.join(
        os.getenv('ROS_HOME', os.path.expanduser('~/.ros')),
        PKG,
        'ycb_video_dataset_mesh_models')
    if not osp.exists(ycb_mesh_models_path):
        gdown.cached_download(
            url='https://drive.google.com/uc?id=1RQ_Ic04P8LrAz28IKbcdatD8tnKbpcYB',  # NOQA
            path=ycb_mesh_models_path + '.tar.gz',
            md5="0ec3c99984027086d454c8ac7a749477",
            postprocess=gdown.extractall,
            quiet=quiet)

    rospack = rospkg.RosPack()
    target_path = osp.join(rospack.get_path(PKG),
                           'sample/data',
                           osp.basename(ycb_mesh_models_path))
    if not osp.exists(target_path):
        os.symlink(ycb_mesh_models_path, target_path)

    ycb_sample_bag_path = osp.join(
        os.getenv('ROS_HOME', os.path.expanduser('~/.ros')),
        PKG,
        'data-0048.bag')
    if not osp.exists(ycb_sample_bag_path):
        gdown.cached_download(
            url='https://drive.google.com/uc?id=1PMPgxD-9Tdu0QKzMnqA_ESPA4UgjsYrn',  # NOQA
            path=ycb_sample_bag_path,
            md5="dbe79664c9e0ab9b11cee7f395ef2a57",
            quiet=quiet)
    target_path = osp.join(rospack.get_path(PKG),
                           'sample/data',
                           osp.basename(ycb_sample_bag_path))
    if not osp.exists(target_path):
        os.symlink(ycb_sample_bag_path, target_path)


if __name__ == '__main__':
    main()
