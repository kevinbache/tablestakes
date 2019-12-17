from tablestakes.fresh import html_css, kv


class Keys:
    key_class = kv.KEY_HTML_CLASS

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

