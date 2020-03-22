import enum
from pathlib import Path
from typing import *

import yattag
import faker
import pandas as pd
from pandas_faker import PandasFaker

from chillpill import params
from python.tablestakes import utils
from python.tablestakes.tryouts import my_css

seed = 1234
fake = faker.Faker()
fake.seed(seed)


from_address_labels = [
    'from',
    'address from',
    'receive from'
    'bill to',
    'bill',
    'billing address',
]

to_address_labels = [
    'to',
    'address to',
    'send to',
    'ship to',
    'shipping',
    'shipping address',
    'sold to',
]


def _make_klass_kwargs(klass_value=None):
    kwargs = {}
    if klass_value is not None:
        kwargs['klass'] = klass_value
    return kwargs


def _add_row_inplace(
        doc,
        values,
        row_tag='tr',
        row_class=None,
        cell_tag='td',
        cell_class=None,
):
    with doc.tag(row_tag, **_make_klass_kwargs(row_class)):
        for val in values:
            with doc.tag(cell_tag, **_make_klass_kwargs(cell_class)):
                doc.text(val)


def df_2_html_table(
        df: pd.DataFrame,
        table_class='my-table',
        do_include_header=True,
        header_class_name='my-table-header',
        row_class_name=None,
        cell_class_name=None,
) -> Text:
    """like df.to_html() but with more control"""
    doc = yattag.Doc()

    with doc.tag('table', klass=table_class):
        if do_include_header:
            _add_row_inplace(
                doc,
                values=df.columns,
                row_tag='thead',
                row_class=header_class_name,
                cell_tag='th',
                cell_class=cell_class_name
            )

        for row_ind, row in df.iterrows():
            _add_row_inplace(
                doc,
                values=row.values,
                row_tag='tr',
                row_class=row_class_name,
                cell_class=cell_class_name,
            )

    return yattag.indent(doc.getvalue())


"""
first we make a hierarchy of tags
"""


def get_address_html(
        container_div_class='address-container',
        label_value='Address:',
        label_div_class='label',
        content_div_class='content',
):
    doc = yattag.Doc()
    # container
    with doc.tag('div', klass=container_div_class):
        # label
        with doc.tag('div', klass=label_div_class):
            doc.text(label_value)

        # content
        with doc.tag('div', klass=content_div_class):
            with doc.tag('address'):
                doc.asis(fake.company() + '<br>\n')
                doc.asis(fake.address().replace('\n', '<br>\n'))

    return yattag.indent(doc.getvalue())


def get_invoice_html(css_str: Text):
    doc, tag, text = yattag.Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('style'):
            doc.asis(f'\n{css_str}')
        with tag('body'):
            # item table
            doc.asis(df_2_html_table(PandasFaker().make_fakes(num_fakes=5), table_class='item-table'))

            doc.asis(get_address_html(container_div_class='address_from', label_value='From:'))
            doc.asis(get_address_html(container_div_class='address_to', label_value='To:'))

    return yattag.indent(doc.getvalue())


def generate_label_box_position_css(
        label_box_config: my_css.LabelBoxConfig,
        container_div_class_name: Text,
) -> my_css.Css:
    num_rows = label_box_config.get_max_rows()
    num_cols = label_box_config.get_max_cols()

    css = my_css.Css()

    css.update_selector(f'.{container_div_class_name}', {
        'display': 'grid',
        'grid-template-columns': ' '.join(['auto'] * num_cols),
        'grid-template-rows': ' '.join(['auto'] * num_rows),
    })

    css.update_selector(f'.{container_div_class_name} .label', {
        'grid-column-start': label_box_config.label_col,
        'grid-row-start': label_box_config.label_row,
    })

    css.update_selector(f'.{container_div_class_name} .content', {
        'grid-column-start': label_box_config.content_col,
        'grid-row-start': label_box_config.content_row,
    })

    return css

"""
Where do things go?
    Header
    Footer
    Main
        top left 
        top right
        lower middle
        grid-area 2437
        anywhere in there

What lines are there?
    Add a class to all that's gridable
    Add another which somehow sets the direction to turn on to get a line between k:v or lines around k:v


What font properties are there?    

Needs to know how to add itself to any container

Add a k:v 

Set k:v style

Let it be flow wrapped in the grid along with all the other stuff.  
say if things need to go before and after each other

"""


class Page:
    pass


class Grid:
    pass


class GridArea:
    """Header, Nav, Section, Article, Aside, Footer like HTML5 + numerics"""
    pass


GridAreaProbabilities = Dict[GridArea, float]


class KeyValueContents:
    """"""
    def __init__(self, k: Text, v: Text):
        self.k = k
        self.v = v




# {css selector: {tag: value}}
CssTemplateDict = Dict[Text, Dict[Text, Text]]

"""
each k/v gets a container class name
    then it's .container .label and .container .contents 
pick template
    colon
    table
    label under fill line
    labelbox (like tax forms)
pick subtemplate
    colon above / colon to the left
    left table / left table
    label    
pick pick details
    bold labels/not
    left/right just
"""

"""
multilevel labels
see printable...ups
"""


class KeyValueTemplate(enum.Enum):
    COLON = {''}
    TABLE = 2
    UNDERLINE = 3
    CELLFILL = 4


class Tag:
    """Div, p, etc."""
    text: Optional[Text]
    klass: Optional[Text]
    id: Optional[Text]
    """
    what goes in properties just for us?
    the middle border
    get class for middle border
    get class for outside label border
    get class for outside content border 
    """
    properties_just_for_us: Dict[Text, Any]

"""
k-v python object
has CSS tags for container, label, content.  container knows how to get border between them  
"""


"""
detect {k: v} style for document
"""


"""
each k/v will need to understand its own variants.  
    eventually we'll find a way to name those states and probabilize them.  
    For example: any table-based k:v pair can have boxes or not. 
    but really that's true for any level of the grid area > div > div hierarchy.
    For each k:v template we need 
    k:v defines an html structure:
        div container
            div label
            div contents
    The relative position of these two things is captured by the LabelBoxConfig. 
        which is really LabelContentsPositions
        Or maybe KeyValuePositions
    Then the question is what other space of things do web-devs vary along
    This is the HTMLGan
    + EventMap classifier
    
    There are transformers we can dump into this stack.  Do or don't 
    This is all the visual problem
    
    html gan
    you've got a vector which is going to generate a tree of html tags
    and fill them with text content 
    
    lstm which can create a tag, or close the top tag on the stack
    detect real or fake html
    
    also just use to classify html
    1) the whole page
        is an event page
    2) a subsection within the page
    There are different rules
        you can't have your text run outside of the current tag
            or can you?
    during generation it is only allowed to drop a closing tag for the latest tag on the stack.  
    then it'll be valid html.

    i want more than just the lstm's linear memory of what things have transpired from above
    but i also want to be able to drag down context directly from my parent or grandparent,
     even if there have been 10k text tokens since i dropped down
     
     there's a separate node in the lstm -- the constant metatdata node -- 
     which is updated when you drop down into that node.  additive     
"""


if __name__ == '__main__':
    position_probabilities = {
        my_css.LabelBoxConfig.UL: 0.3,
        my_css.LabelBoxConfig.U: 1.0,
        my_css.LabelBoxConfig.UR: 0.01,
        my_css.LabelBoxConfig.R: 0.1,
        my_css.LabelBoxConfig.BR: 0.01,
        my_css.LabelBoxConfig.B: 0.2,
        my_css.LabelBoxConfig.BL: 0.01,
        my_css.LabelBoxConfig.L: 1.0,
    }


    class InvoiceGenerationParams(params.ParameterSet):
        # do_include_header = True
        # do_include_footer = True
        label_position = params.Categorical.from_prob_dict(position_probabilities)
        # probability that we right justify if the label is to the left or above-or-below the content
        label_rjust_prob = 0.5


    # normalize the probabilities
    s = sum(position_probabilities.values())
    for k, v in position_probabilities.items():
        position_probabilities[k] = v / s

    for config in my_css.LabelBoxConfig:
        css = generate_label_box_position_css(config, 'address_to')
        css += generate_label_box_position_css(config, 'address_from')

        html = get_invoice_html(str(css))
        out_dir = Path('./output')
        out_dir.mkdir(exist_ok=True)

        utils.save_txt(html, out_dir / f'out_{config.name}.html')
