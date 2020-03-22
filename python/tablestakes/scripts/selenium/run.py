from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from python.tablestakes import utils


options = Options()
# options.headless = True
dpi = 400
window_width_px = dpi * 8.5
window_height_px = dpi * 11.00

with utils.Timer('driver='):
    driver = webdriver.Firefox(options=options)
    driver.set_window_position(0, 0)
    driver.set_window_size(window_width_px, window_height_px)
with utils.Timer('driver.get'):
    # driver.get("http://google.com/")
    driver.get("file:///Users/kevin/projects/tablestakes/tablestakes/scripts/generate_ocrd_doc/doc_wrapped.html")
with utils.Timer('execute script'):
    script = """
    var w = document.getElementById("word_000000");
    return w.getBoundingClientRect();
    """
    out = driver.execute_script(script)
    print(f'execute out: {out}')
with utils.Timer('driver.quit'):
    driver.quit()


"""
function findPos(obj) {
    var curleft = curtop = 0;
    if (obj.offsetParent) {
        do {
            curleft += obj.offsetLeft;
            curtop += obj.offsetTop;
        } while (obj = obj.offsetParent);
    }
}
var w = document.getElementById("demo");
w.getBoundingClientRect();
"""

"""
https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
Element.getBoundingClientRect()
add in window.scrollX and window.scrollY for page position
"""

"""
window.devicePixelRatio
2
"""

"""
function getDPI() {
    var div = document.createElement( "div");
    div.style.height = "1in";
    div.style.width = "1in";
    div.style.top = "-100%";
    div.style.left = "-100%";
    div.style.position = "absolute";

    document.body.appendChild(div);

    var result =  div.offsetHeight;

    document.body.removeChild( div );

    return result;

}
"""

