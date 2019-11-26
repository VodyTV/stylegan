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
import tqdm
import dnnlib
import dnnlib.tflib as tflib
import config


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    dataset_urls = dict(
        faces='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
        cats='https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ',
        bedrooms='https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF',
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

    num_batches = int(np.ceil(len(features_df) / args.batch_size))
    for batch_i in tqdm.tqdm(list(range(num_batches))):
        batch_offset = batch_i * args.batch_size
        batch_end = min(batch_offset + args.batch_size, len(features_df))
        indexes = range(batch_offset, batch_end)
        rows = features_df.iloc[indexes]
        features = np.vstack(rows['features'].to_numpy()).astype(np.float32)
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
        for wid, image in zip(rows['watchable_id'], images):
            png_filename = os.path.join(config.result_dir, f'{wid}.png')
            img = PIL.Image.fromarray(image, 'RGB')
            if args.output_size:
                img = img.resize(
                    (args.output_size, args.output_size),
                    PIL.Image.ANTIALIAS
                )
            img.save(png_filename)


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
    features.reset_index(inplace=True, drop=True)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path')
    parser.add_argument('--dataset', default='cats')
    parser.add_argument('--truncation', type=float, default=1.)
    parser.add_argument('--scaling', default='normalize')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-size', type=int, default=256)
    parser.add_argument('--wids', default=None, type=int, nargs='+',
                            help='Render only these watchables by id.')
    args = parser.parse_args()

    main(args)
