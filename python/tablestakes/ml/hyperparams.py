import hashlib

import numpy as np

from chillpill import params

from tablestakes import constants, kv_styles
from tablestakes import html_css as hc


class LearningParams(params.ParameterSet):
    ##############
    # model
    #  embedder
    num_embedding_dim = 12
    do_include_embeddings = True

    #  transformer
    pre_trans_linear_dim = None

    num_trans_enc_layers = 4
    num_trans_heads = 8
    num_trans_fc_dim_mult = 4

    do_cat_x_base_before_fc = True

    #  fully connected
    num_fc_blocks = 4
    log2num_neurons_start = 5
    log2num_neurons_end = 5
    num_fc_blocks_per_resid = 2

    num_fc_layers_per_dropout = 10
    # prob of dropping each unit
    dropout_p = 0.5

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
    asha_reduction_factor = 4

    ##############
    # data
    # batch size must be 1
    log2_batch_size = 5
    p_valid = 0.1
    p_test = 0.1
    data_dir = constants.DOCS_DIR / 'num=2000_e4d0'
    dataset_file = constants.DOCS_DIR / 'ds_num=2000_e4d0.cloudpickle'

    # for transferring objects from tune process to worker processes
    ds_ray_obj_id = None
    num_y_classes_ray_obj_id = None
    word_to_id_ray_obj_id = None
    word_to_count_ray_obj_id = None

    # for data loading
    num_workers = 4

    ##############
    # extra
    num_steps_per_histogram_log = 100

    upload_dir = 's3://kb-tester-2020-10-08'
    project_name = 'tablestakes_trans1d_tests'

    num_gpus = 1

    seed = 42


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

    seed_start = 42

    num_docs = 1000

    doc_gen_params = DocGenParams()
    doc_prep_params = DocPrepParams()

    docs_root_dir = constants.DOCS_DIR
    docs_dir = None

    def _get_short_hash(self, num_chars=4):
        d = self.to_dict()
        h = hashlib.sha1(str(d).encode('utf-8'))
        return h.hexdigest()[-num_chars:]

    def set_docs_dir(self):
        self.docs_dir = \
            self.docs_root_dir / f'num={self.num_docs}_{self._get_short_hash()}'
