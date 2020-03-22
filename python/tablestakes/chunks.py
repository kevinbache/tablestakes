from python.tablestakes import html_css
from python.tablestakes.html_css import SelectorType


class Keys:
    key_class = SelectorType.KEY.html_class_name

    bold = html_css.CssChunk(
        f'.{key_class}', {
            # 'border-bottom-style': 'solid',
            # 'border-bottom-width': 1,
            'font-weight': 'bold',
        },
    )

    add_colon = html_css.CssChunk(
        f'.{key_class}:after', {
            'content': "':'",
        }
    )


class Body:
    # https://www.w3.org/Style/Examples/007/fonts.en.html
    font_sans_serif = html_css.CssChunk('body', {
        'font': 'normal 12px Verdana, Arial, sans-serif',
    })

    font_serif = html_css.CssChunk('body', {
        'font': 'normal 12px Times, Times New Roman, Georgia, serif',
    })

    font_mono = html_css.CssChunk('body', {
        'font': 'normal 12px Courier New, monospace',
    })

    jelly_legs = html_css.CssChunk('body', {
        'lower-style': 'squiggle jelly underline',
    })
