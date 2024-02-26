from __future__ import print_function

import configargparse


def model_opts(parser):

    group = parser.add_argument_group('Model AE')

def extract_opts(parser):

    group = parser.add_argument_group('General')
    group.add('--src_dir', '-src_dir', type=str, required=True,
              help="System path to the Samson directory.")

    group.add('--num_bands', '-num_bands', type=int, default=156,
              help="Number of spectral bands present in input image.")

    group.add('--end_members', '-end_members', type=int, default=3,
              help="Number of end-members to be extracted from HSI.")

    group = parser.add_argument_group('Hyperparameters')
    group.add('--batch_size', '-batch_size', type=int, default=5,
              help="Maximum batch size for training.")

    group.add('--training', '-training', type=int, default=5,
              help="Maximum batch size for training.")

    group.add('--time', '-time', type=float, default=1.0,
              help="Defines the time for the training.")

    group.add('--image_size', '-image_size', type=int, default=100,
              help="Defines the size of image.")
    group.add('--patch_size', '-patch_size', type=int, default=15,
              help="Defines the size of image.")
   