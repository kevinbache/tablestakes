# import hashlib
# from typing import *
#
# import numpy as np
#
# from chillpill import new_properties
#
# from tablestakes import constants, kv_styles, utils
# from tablestakes import html_css as hc
#
#
# class LearningParams(new_properties.ParameterSet):
#     def __init__(
#             self,
#             dataset_name: str,
#             docs_dir_base=constants.DOCS_DIR,
#             datasets_dir=constants.DATASETS_DIR,
#             **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.dataset_name = dataset_name
#         self.docs_dir_base = docs_dir_base
#         self.datasets_dir = datasets_dir
#
#         self.docs_dir = None
#         self.dataset_file = None
#         self.update_files()
#
#     def update_files(self):
#         self.docs_dir = self.docs_dir_base / self.dataset_name
#         self.dataset_file = self.datasets_dir / f'{self.dataset_name}.cloudpickle'
#
#     def get_batch_size(self):
#         return utils.pow2int(self.log2_batch_size)
#
#     # data
#
#     #  embedder
#     do_include_embeddings = True
#     num_embedding_base_dim = 64
#     num_extra_embedding_dim = None
#     # embed = param_torch_mods.BertEmbedder.DataParams()
#     # embed.dim = None
#     # embed.requires_grad = True
#
#     #  transformer
#     pre_trans_linear_dim = None
#
#     num_trans_enc_layers = 4
#     num_trans_heads = 8
#     num_trans_fc_dim_mult = 4
#
#     trans_encoder_type = 'torch'
#
#     do_cat_x_base_before_fc = True
#
#     #  fully connected
#     # num_fc_blocks = 4
#     # log2num_neurons_start = 5
#     # log2num_neurons_end = 5
#     # num_fc_blocks_per_resid = 2
#
#     # fc = param_torch_mods.SlabNet.DataParams(
#     #     num_features=32,
#     #     num_layers=2,
#     #     num_groups=32,
#     #     num_blocks_per_residual=2,
#     #     do_include_first_norm=True,
#     # )
#
#     # num_fc_layers_per_dropout = 10
#     # # prob of dropping each unit
#     # dropout_p = 0.5
#
#     #  head_nets
#     # num_head_blocks = 2
#     # log2num_head_neurons = 4
#     # num_head_blocks_per_resid = 1
#
#     # sub_losses = param_torch_mods.HeadedSlabNet.DataParams(
#     #     num_features=32,
#     #     num_layers=2,
#     #     num_groups=32,
#     #     num_blocks_per_residual=1,
#     # )
#
#     ##############
#     # optimization
#     lr = 0.001
#
#     # korv, which_kv
#     korv_loss_weight = 0.5
#
#     num_epochs = 4
#
#     ##############
#     # search_params search
#     num_hp_samples = 100
#     search_metric = 'valid_acc_which_kv'
#     search_mode = 'max'
#     asha_grace_period = 4
#     asha_reduction_factor = 2
#
#     ##############
#     # data
#     # batch size must be 1
#     log2_batch_size = 5
#     p_valid = 0.1
#     p_test = 0.1
#     dataset_name = 'num=10_c145'
#
#     max_seq_length = int(np.power(2, 14))
#
#     # for data loading
#     num_workers = 4
#
#     ##############
#     # extra
#     num_steps_per_metric_log = 100
#     num_steps_per_histogram_log = 500
#
#     logs_dir = constants.OUTPUT_DIR
#     upload_dir = 's3://kb-tester-2020-10-14'
#     # neptune won't let you create projects from its api so this has to already exist
#     project_name = 'tablestakes'
#     experiment_name = 'trans_v0.1.3'
#     group_name = 'log2_batch'
#
#     experiment_tags = ['default', 'testing']
#
#     num_cpus = 2
#     num_gpus = 1
#
#     seed = 42
#
#     def get_project_exp_name(self):
#         return f'{self.project_name}_{self.experiment_name}'
#
#     def get_exp_group_name(self):
#         return f'{self.experiment_name}_{self.group_name}'
#
#
# class DocGenParams(new_properties.ParameterSet):
#     margin = '0.5in'
#     page_size = hc.PageSize.LETTER
#     dpi = new_properties.Discrete.from_prob_dict({
#         300: 1,
#         250: 1,
#     })
#
#     do_randomize_field_order = True
#
#     group_offset_in = new_properties.Float(0.0, 3.0)
#     do_set_invisible_border = new_properties.Boolean(p_true=0.9)
#     num_extra_fields = new_properties.Discrete.from_prob_dict({
#         0: 1,
#         1: 1,
#     })
#     font_size_px = new_properties.Integer(8, 19)
#     val_font_size_px = new_properties.Integer(8, 19)
#     do_regen_font_val_size = new_properties.Boolean(0.4)
#     table_cell_padding_px = new_properties.Integer(1, 7)
#     do_bold_keys = new_properties.Boolean(p_true=0.2)
#     do_add_colon_to_keys = new_properties.Boolean(p_true=0.2)
#     vert_alignment = new_properties.Categorical.from_prob_dict({
#         kv_styles.KvAlign.TT: 2,
#         kv_styles.KvAlign.BB: 1,
#         kv_styles.KvAlign.TB: 1,
#     })
#     horz_alignment = new_properties.Categorical.from_prob_dict({
#         kv_styles.KvAlign.LL: 2,
#         kv_styles.KvAlign.RL: 1,
#         kv_styles.KvAlign.LR: 1,
#         kv_styles.KvAlign.CL: 0.2,
#         kv_styles.KvAlign.LC: 0.2,
#     })
#
#     def get_page_size_px(self):
#         return [self.dpi * self.page_size.width, self.dpi * self.page_size.height]
#
#
# class DocPrepParams(new_properties.ParameterSet):
#     min_count_to_keep_word = 4
#
#
# class DocSetParams(new_properties.ParameterSet):
#     def __init__(
#             self,
#             doc_gen_params: DocGenParams = DocGenParams(),
#             doc_prep_params: DocPrepParams = DocPrepParams(),
#             **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.doc_gen_params = doc_gen_params
#         self.doc_prep_params = doc_prep_params
#         self._short_hash = None
#
#     seed_start = 1234
#
#     num_docs = 10
#
#     doc_gen_params = DocGenParams()
#     doc_prep_params = DocPrepParams()
#
#     _data_dir = constants.DATA_DIR
#     docs_root_dir = constants.DOCS_DIR
#     dataset_root_dir = constants.DATASETS_DIR
#
#     docs_dir = None
#
#     def get_doc_set_name(self):
#         return f'num={self.num_docs}_{self.get_short_hash()}'
#
#     def get_dataset_dir(self):
#         return self.docs_root_dir / 'dataset'
#
#     def get_dataset_file(self):
#         return self.dataset_root_dir / f'{self.get_doc_set_name()}.cloudpickle'
#
#     def get_docs_dir(self):
#         return self.docs_root_dir / self.get_doc_set_name()
#
#     def set_docs_dir(self):
#         self.docs_dir = self.get_docs_dir()

import numpy as np

from chillpill import params

from tablestakes import constants, utils
from tablestakes.create_fake_data import kv_styles, html_css as hc

#
# class LearningParams(params.ParameterSet):
#     def __init__(
#             self,
#             dataset_name: str,
#             docs_dir_base=constants.DOCS_DIR,
#             datasets_dir=constants.DATASETS_DIR,
#             **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.dataset_name = dataset_name
#         self.docs_dir_base = docs_dir_base
#         self.datasets_dir = datasets_dir
#
#         self.docs_dir = None
#         self.dataset_file = None
#         self.update_files()
#
#     def update_files(self):
#         self.docs_dir = self.docs_dir_base / self.dataset_name
#         self.dataset_file = self.datasets_dir / f'{self.dataset_name}.cloudpickle'
#
#     def get_batch_size(self):
#         return utils.pow2int(self.log2_batch_size)
#
#     # data
#
#     #  embedder
#     do_include_embeddings = True
#     num_embedding_base_dim = 64
#     num_extra_embedding_dim = None
#     # embed = param_torch_mods.BertEmbedder.DataParams()
#     # embed.dim = None
#     # embed.requires_grad = True
#
#     #  transformer
#     pre_trans_linear_dim = None
#
#     num_trans_enc_layers = 4
#     num_trans_heads = 8
#     num_trans_fc_dim_mult = 4
#
#     trans_encoder_type = 'torch'
#
#     do_cat_x_base_before_fc = True
#
#     #  fully connected
#     # num_fc_blocks = 4
#     # log2num_neurons_start = 5
#     # log2num_neurons_end = 5
#     # num_fc_blocks_per_resid = 2
#
#     # fc = param_torch_mods.SlabNet.DataParams(
#     #     num_features=32,
#     #     num_layers=2,
#     #     num_groups=32,
#     #     num_blocks_per_residual=2,
#     #     do_include_first_norm=True,
#     # )
#
#     # num_fc_layers_per_dropout = 10
#     # # prob of dropping each unit
#     # dropout_p = 0.5
#
#     #  head_nets
#     # num_head_blocks = 2
#     # log2num_head_neurons = 4
#     # num_head_blocks_per_resid = 1
#
#     # sub_losses = param_torch_mods.HeadedSlabNet.DataParams(
#     #     num_features=32,
#     #     num_layers=2,
#     #     num_groups=32,
#     #     num_blocks_per_residual=1,
#     # )
#
#     ##############
#     # optimization
#     lr = 0.001
#
#     # korv, which_kv
#     korv_loss_weight = 0.5
#
#     num_epochs = 4
#
#     ##############
#     # search_params search
#     num_hp_samples = 100
#     search_metric = 'valid_acc_which_kv'
#     search_mode = 'max'
#     asha_grace_period = 4
#     asha_reduction_factor = 2
#
#     ##############
#     # data
#     # batch size must be 1
#     log2_batch_size = 5
#     p_valid = 0.1
#     p_test = 0.1
#     dataset_name = 'num=10_c145'
#
#     max_seq_length = int(np.power(2, 14))
#
#     # for data loading
#     num_workers = 4
#
#     ##############
#     # extra
#     num_steps_per_metric_log = 100
#     num_steps_per_histogram_log = 500
#
#     logs_dir = constants.OUTPUT_DIR
#     upload_dir = 's3://kb-tester-2020-10-14'
#     # neptune won't let you create projects from its api so this has to already exist
#     project_name = 'tablestakes'
#     experiment_name = 'trans_v0.1.3'
#     group_name = 'log2_batch'
#
#     experiment_tags = ['default', 'testing']
#
#     num_cpus = 2
#     num_gpus = 1
#
#     seed = 42
#
#     def get_project_exp_name(self):
#         return f'{self.project_name}_{self.experiment_name}'
#
#     def get_exp_group_name(self):
#         return f'{self.experiment_name}_{self.group_name}'


class DocGenParams(params.ParameterSet):
    margin = '0.5in'
    page_size = hc.PageSize.LETTER
    dpi = params.Discrete.from_prob_dict({
        300: 1,
        250: 1,
        200: 1,
    })

    do_randomize_field_order = True

    group_offset_in = params.Float(0.0, 3.0)
    do_set_invisible_border = params.Boolean(p_true=0.9)
    num_extra_fields = params.Discrete.from_prob_dict({
        0: 1,
        1: 1,
    })
    font_size_px = params.Integer(8, 19)
    val_font_size_px = params.Integer(8, 19)
    do_regen_font_val_size = params.Boolean(0.4)
    table_cell_padding_px = params.Integer(1, 7)
    do_bold_keys = params.Boolean(p_true=0.2)
    do_add_colon_to_keys = params.Boolean(p_true=0.2)
    vert_alignment = params.Categorical.from_prob_dict({
        kv_styles.KvAlign.TT: 2,
        kv_styles.KvAlign.BB: 1,
        kv_styles.KvAlign.TB: 1,
    })
    horz_alignment = params.Categorical.from_prob_dict({
        kv_styles.KvAlign.LL: 2,
        kv_styles.KvAlign.RL: 1,
        kv_styles.KvAlign.LR: 1,
        kv_styles.KvAlign.CL: 0.2,
        kv_styles.KvAlign.LC: 0.2,
    })

    def get_page_size_px(self):
        return [self.dpi * self.page_size.width, self.dpi * self.page_size.height]


class DocPrepParams(params.ParameterSet):
    min_count_to_keep_word = 4


class DocSetParams(params.ParameterSet):
    def __init__(
            self,
            doc_gen_params: DocGenParams = DocGenParams(),
            doc_prep_params: DocPrepParams = DocPrepParams(),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc_gen_params = doc_gen_params
        self.doc_prep_params = doc_prep_params
        self._short_hash = None

    seed_start = 1234

    num_docs = 10

    doc_gen_params = DocGenParams()
    doc_prep_params = DocPrepParams()

    _data_dir = constants.DATA_DIR
    docs_root_dir = constants.DOCS_DIR
    dataset_root_dir = constants.DATASETS_DIR

    docs_dir = None

    def get_doc_set_name(self):
        return f'num={self.num_docs}_{self.get_short_hash()}'

    def get_dataset_dir(self):
        return self.docs_root_dir / 'dataset'

    def get_dataset_file(self):
        return self.dataset_root_dir / f'{self.get_doc_set_name()}.cloudpickle'

    def get_docs_dir(self):
        return self.docs_root_dir / self.get_doc_set_name()

    def set_docs_dir(self):
        self.docs_dir = self.get_docs_dir()

