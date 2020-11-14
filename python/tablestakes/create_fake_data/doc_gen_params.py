
from chillpill import params

from tablestakes import constants, utils
from tablestakes.create_fake_data import html_css as hc, kv_styles


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

