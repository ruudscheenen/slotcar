import pyvisual as pv
from ui.ui_page_0 import create_page_0_ui


def create_window():
    window = pv.PvWindow(
        title="PyVisual Window",
        width=1920,
        height=1080,
        bg_color=(255, 255, 255, 1),
        icon=None,
        bg_image="assets/background/background.jpg",
        is_frameless=False,
        is_resizable=False
    )
    return window


def create_pages(window):
    pages = pv.PvPages(window, animation_duration=0, animation_orientation="horizontal")
    page_0 = pages.create_page("page_0",   bg_color=(255, 255, 255, 1),  bg_image="assets/background/background.jpg")
    return pages, page_0


def create_ui():
    window = create_window()
    pages, page_0 = create_pages(window)
    page_0_widget = pages.widget(page_0)
    ui = {
        "window": window,
        "pages": pages
    }
    ui_page_0 = create_page_0_ui(page_0_widget, ui)

    ui.update({
        "page_0": ui_page_0
    })

    return ui
