from tablestakes import constants
import numpy as np

from chillpill import params


class MyHyperparams(params.ParameterSet):
    ##############
    # model
    #  conv
    num_conv_layers = 6
    log2num_filters_start = 5
    log2num_filters_end = 4

    kernel_size = 7

    num_conv_layers_per_pool = 2
    pool_size = 2

    #  fc
    num_fc_hidden_layers = 3
    log2num_neurons_start = 6
    log2num_neurons_end = 5

    num_fc_layers_per_dropout = 2
    dropout_p = 0.5

    num_embedding_dim = 32
    do_include_embeddings = False

    ##############
    # optimization
    lr = 0.001

    # doesn't entirely work cause vocab recalc...
    limit_num_data = None

    # korv, which_kv
    loss_weights = np.array([1.0, 0.0])

    num_epochs = 5

    ##############
    # data

    # batch size must be 1
    batch_size_log2 = 0
    p_valid = 0.1
    p_test = 0.1
    data_dir = constants.DOCS_DIR

    # for data loading
    num_workers = 2
