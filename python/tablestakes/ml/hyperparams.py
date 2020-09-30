from tablestakes import constants
import numpy as np

from chillpill import params


class MyHyperparams(params.ParameterSet):
    ##############
    # model
    #  embedder
    num_embedding_dim = 15
    do_include_embeddings = True

    #  transformer
    pre_trans_linear_dim = 16

    num_trans_enc_layers = 2
    num_trans_heads = 8
    num_trans_fc_units = 2048

    do_include_batch_norm = True

    #  fully connected
    num_fc_hidden_layers = 3
    log2num_neurons_start = 5
    log2num_neurons_end = 5

    num_fc_layers_per_dropout = 1
    # prob of drop, not prob of keep
    dropout_p = 0.5

    ##############
    # optimization
    lr = 0.001

    # doesn't entirely work cause vocab recalc...
    limit_num_data = None

    # korv, which_kv
    loss_weights = np.array([0.0, 1.0])
    # loss_weights = np.array([1.0, 0.0])

    num_epochs = 50

    ##############
    # data

    # batch size must be 1
    batch_size_log2 = 0
    p_valid = 0.1
    p_test = 0.1
    data_dir = constants.DOCS_DIR

    # for data loading
    num_workers = 2
