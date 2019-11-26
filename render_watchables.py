# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import PIL.Image
from sklearn import preprocessing
import dnnlib
import dnnlib.tflib as tflib
import config


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    dataset_urls = dict(
        faces='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
        cats='https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ'
    )
    url = dataset_urls[args.dataset]

    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    features_df = load_features(args)
    # potentially render only requested watchables
    if args.wids:
        wids = set(args.wids)
        features_df = features_df[features_df['watchable_id'].isin(wids)]

    os.makedirs(config.result_dir, exist_ok=True)

    for row_id, row in features_df.iterrows():
        wid = row['watchable_id']
        print('rendering', wid)
        features = row['features'].astype(np.float32)[None, :]

        # Pick latent vector.
        z_dims = Gs.input_shape[1]
        num_features, feature_dims = features.shape
        # Tile our feature vector until it has the required dimensionality
        missing_dims = z_dims - feature_dims
        num_copies = int(np.ceil(missing_dims / feature_dims)) + 1
        z_features = np.tile(features, (1, num_copies))[:, :z_dims]
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(
            z_features, None, truncation_psi=0.7, randomize_noise=True,
            output_transform=fmt
        )

        # Save image.
        png_filename = os.path.join(config.result_dir, f'{wid}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


def load_features(args):
    features = pd.read_pickle(args.features_path)
    features.rename(dict(item_feature_id='watchable_id'), inplace=True, axis=1)
    if args.scaling != 'none':
        features_arr = np.vstack(features['features'].values)
        if args.scaling == 'std':
            scaler = preprocessing.StandardScaler().fit(features_arr)
        elif args.scaling == 'maxabs':
            scaler = preprocessing.MaxAbsScaler().fit(features_arr)
        elif args.scaling == 'normalize':
            scaler = preprocessing.Normalizer().fit(features_arr)
        elif args.scaling == 'clip':
            scaler = preprocessing.FunctionTransformer(
                lambda x: x.clip(-args.truncation, args.truncation)
            )
        features['features'] = list(scaler.transform(features_arr))
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path')
    parser.add_argument('--dataset', default='cats')
    parser.add_argument('--truncation', type=float, default=1.)
    parser.add_argument('--scaling', default='normalize')
    parser.add_argument('--wids', default=None, type=int, nargs='+',
                            help='Render only these watchables by id.')
    args = parser.parse_args()

    main(args)
