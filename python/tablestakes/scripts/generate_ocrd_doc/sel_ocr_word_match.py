from typing import Dict

from lxml import etree
import numpy as np
import pandas as pd
from selenium import webdriver

from tablestakes import doc as ts_doc


# TODO: these conversion functions are ugly


def selenium_row_to_bbox(srow: pd.Series):
    print(f"selenium_row_to_bbox srow: {srow}")
    return ts_doc.BBox(
        left=float(srow['left']),
        right=float(srow['right']),
        top=float(srow['top']),
        bottom=float(srow['bottom']),
    )


def get_line(ocr_points: np.array, selenium_points: np.array):
    """
    where
      ocr_points = np.array([ocr_bbox.xmin, orc_box.xmax])
      selenium_points = np.array([sel_bbox.xmin, sel_box.xmax])

    can be x or y
    """
    assert ocr_points.shape == (2,)
    assert selenium_points.shape == (2,)

    m = (ocr_points[0] - ocr_points[1]) / (selenium_points[0] - selenium_points[1])

    b = (m * selenium_points - ocr_points)
    assert np.abs(b[0] - b[0]) < 0.01
    b = b[0]

    return np.array([m, b])


def get_lines(ocr_bbox: ts_doc.BBox, selenium_bbox: ts_doc.BBox):
    x_line = get_line(
        np.array([ocr_bbox.left, ocr_bbox.right]),
        np.array([selenium_bbox.left, selenium_bbox.right]),
    )

    y_line = get_line(
        np.array([ocr_bbox.top, ocr_bbox.bottom]),
        np.array([selenium_bbox.top, selenium_bbox.bottom]),
    )

    return x_line, y_line


def _convert_point(line: np.array, point: float):
    assert line.shape == (2,)
    return line[0] * point - line[1]


def convert_bbox(x_line: np.array, y_line: np.array, sel_bbox: ts_doc.BBox):
    return ts_doc.BBox(
        left=_convert_point(x_line, sel_bbox.left),
        right=_convert_point(x_line, sel_bbox.right),
        top=_convert_point(y_line, sel_bbox.top),
        bottom=_convert_point(y_line, sel_bbox.bottom),
    )


def get_word_pixel_locations(
        html_file: str,
        word_id_to_word: Dict[str, etree._Element],
        window_width_px: float = 400 * 8.5,
        window_height_px: float = 400 * 11.0,
) -> pd.DataFrame:
    options = webdriver.firefox.options.Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    try:
        driver.set_window_position(0, 0)
        driver.set_window_size(window_width_px, window_height_px)
        driver.get(html_file)

        word_locations = []
        for word_id, word in word_id_to_word.items():
            script = f"""
            var w = document.getElementById("{word_id}");
            var rect = w.getBoundingClientRect();
            return rect;
            """

            """
            Example: 
            word_location == {
                'left': 12,
                'right': 58.75,
                'top': 12,
                'bottom': 33,
                'x': 12,
                'width': 46.75,
                'y': 12,
                'height': 21,
            }
            """
            word_location = driver.execute_script(script)
            word_location['word_id'] = word_id
            word_location['text'] = word.text
            word_locations.append(word_location)
    finally:
        driver.quit()

    return pd.DataFrame(word_locations)

