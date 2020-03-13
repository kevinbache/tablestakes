from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from tablestakes import utils


options = Options()
# options.headless = True
window_width_px = 300 * 8.5
window_height_px = 300 * 11.00

with utils.Timer('driver='):
    driver = webdriver.Firefox(options=options)
    driver.set_window_position(0, 0)
    driver.set_window_size(window_width_px, window_height_px)
with utils.Timer('driver.get'):
    # driver.get("http://google.com/")
    driver.get("file://")
with utils.Timer('execute script'):
    out = driver.execute_script("return true")
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
var obj = document.getElementById("demo");
return findPos(obj);
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

