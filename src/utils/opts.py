from __future__ import print_function

import configargparse


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group('Model AE')
    group.add('--encoder_type', '-encoder_type', type=str, default='deep',
              choices=['deep', 'shallow'],
              help="Allows the user to choose between two levels of encoder complexity."
                   "Options are: [deep|shallow]")

    # SLReLU unavailable, add assert in main
    group.add('--soft_threshold', '-soft_threshold', type=str, default='SReLU',
              choices=['SReLU', 'SLReLU'],
              help="Type of soft-thresholding for final layer of encoder"
                   "Options are: [SReLU|SLReLU]")

    group.add('--activation', '-activation', type=str,
              choices=['ReLU', 'Leaky-ReLU', 'Sigmoid'],
              help="Activation function for hidden layers of encoder."
                   "For shallow AE there won't be any activation. Options are:"
                   "[ReLU|Leaky-ReLU|Sigmoid]")


def train_opts(parser):
    """
    These options are passed to the training of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group('General')
    group.add('--src_dir', '-src_dir', type=str, default="../data/Samson",
              help="System path to the Samson directory.")

    group.add('--num_bands', '-num_bands', type=int, default=156,
              help="Number of spectral bands present in input image.")

    group.add('--end_members', '-end_members', type=int, default=3,
              help="Number of end-members to be extracted from HSI.")

    group = parser.add_argument_group('Hyperparameters')
    group.add('--batch_size', '-batch_size', type=int, default=5,
              help="Maximum batch size for training.")

    group.add('--learning_rate', '-learning_rate', type=float, default=1e-3,
              help="Learning rate for training the network.")

    group.add('--epochs', '-epochs', type=int, default=80,
              help="Number of iterations that the network should be trained for.")
    
    group.add('--training', '-training', type=str, default='True',
              help="Defines the threshold for the soft-thresholding operation.")
    group.add('--patch_size', '-patch_size', type=int, default=5,
              help="Defines the threshold for the soft-thresholding operation.")