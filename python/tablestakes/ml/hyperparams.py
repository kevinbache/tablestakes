import hashlib
from typing import *

import numpy as np

from chillpill import params

from tablestakes import constants, kv_styles
from tablestakes import html_css as hc


class LearningParams(params.ParameterSet):
    def __init__(self, dataset_name: str, **kwargs):
        super().__init__(**kwargs)

        self.dataset_name = dataset_name
        self.docs_dir = None
        self.dataset_file = None
        self.update_files()

    def update_files(self):
        self.docs_dir = constants.DOCS_DIR / self.dataset_name
        self.dataset_file = constants.DATASETS_DIR / f'{self.dataset_name}.cloudpickle'

    ##############
    # model
    #  embedder
    num_embedding_dim = 12
    do_include_embeddings = True
    num_extra_embedding_dim = None

    #  transformer
    pre_trans_linear_dim = None

    num_trans_enc_layers = 4
    num_trans_heads = 8
    num_trans_fc_dim_mult = 4

    trans_encoder_type = 'torch'

    do_cat_x_base_before_fc = True

    #  fully connected
    num_fc_blocks = 4
    log2num_neurons_start = 5
    log2num_neurons_end = 5
    num_fc_blocks_per_resid = 2

    num_fc_layers_per_dropout = 10
    # prob of dropping each unit
    dropout_p = 0.5

    #  head_nets
    num_head_blocks = 2
    log2num_head_neurons = 4
    num_head_blocks_per_resid = 1

    # https://arxiv.org/pdf/1803.08494.pdf
    # paper default is 32
    # optimal num channels for group is 16
    # conv block maker will min this to num_neurons // 4
    num_groups_for_gn = 32

    ##############
    # optimization
    lr = 0.001

    # korv, which_kv
    korv_loss_weight = 0.5

    num_epochs = 4

    ##############
    # hp search
    num_hp_samples = 100
    search_metric = 'valid_acc_which_kv'
    search_mode = 'max'
    asha_grace_period = 4
    asha_reduction_factor = 2

    ##############
    # data
    # batch size must be 1
    log2_batch_size = 5
    p_valid = 0.1
    p_test = 0.1
    dataset_name = 'num=10_c145'

    # for data loading
    num_workers = 4

    ##############
    # extra
    num_steps_per_metric_log = 100
    num_steps_per_histogram_log = 500

    logs_dir = constants.LOGS_DIR
    upload_dir = 's3://kb-tester-2020-10-14'
    # neptune won't let you create projects from its api so this has to already exist
    project_name = 'tablestakes'
    experiment_name = 'trans_v0.1.3'
    group_name = 'log2_batch'

    experiment_tags = ['default', 'testing']

    num_cpus = 2
    num_gpus = 1

    seed = 42

    @classmethod
    def from_dict(cls, d: Dict):
        hp = cls(dataset_name='dataset_file')
        for k, v in d.items():
            # TODO: don't cast array to int
            if np.issubdtype(type(v), np.integer):
                v = int(v)
            elif np.issubdtype(type(v), np.floating):
                v = float(v)
            hp.__setattr__(k, v)
        return hp

    def get_project_exp_name(self):
        return f'{self.project_name}_{self.experiment_name}'

    def get_exp_group_name(self):
        return f'{self.experiment_name}_{self.group_name}'


class DocGenParams(params.ParameterSet):
    margin = '0.5in'
    page_size = hc.PageSize.LETTER
    dpi = params.Discrete.from_prob_dict({
        300: 1,
        250: 1,
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
    min_count_to_keep_word = 2


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

    seed_start = 42

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

