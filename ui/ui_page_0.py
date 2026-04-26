import pyvisual as pv


def create_page_0_ui(window,ui):
    """
    Create and return UI elements for Page 0.
    :param container: The page widget for Page 0.
    :return: Dictionary of UI elements.
    """
    ui_page = {}
    ui_page["Image_0"] = pv.PvImage(container=window, x=787, y=-7, image_path='assets/images/347ef8f506.jpg',
        scale=1.775, corner_radius=0, flip_v=False, flip_h=False,
        rotate=0, border_color=None, border_hover_color=None, on_hover=None,
        on_click=None, on_release=None, border_thickness=0, border_style="solid",
        is_visible=True, opacity=1, tag=None)

    ui_page["Slider_1"] = pv.PvSlider(container=window, x=938, y=347, width=230,
        height=50, min_value=0, max_value=100, value=80,
        track_color=(200, 200, 200, 1), track_border_color=(180, 180, 180, 1), fill_color=(219, 0, 255, 1), knob_color=(219, 0, 255, 1),
        knob_border_color=(255, 255, 255, 1), track_corner_radius=2, knob_corner_radius=11, knob_width=20,
        knob_height=20, knob_size=5, show_text=False, value_text=0,
        min_text=0, max_text=100, font='assets/fonts/Poppins/Poppins.ttf', font_size=12,
        font_color=(0, 0, 0, 1), bold=False, italic=False, underline=False,
        strikethrough=False, opacity=1, track_border_thickness=0, knob_border_thickness=3,
        track_height=10, is_visible=True, is_disabled=False)

    ui_page["Slider_2"] = pv.PvSlider(container=window, x=938, y=38, width=230,
        height=50, min_value=0, max_value=100, value=80,
        track_color=(200, 200, 200, 1), track_border_color=(180, 180, 180, 1), fill_color=(219, 0, 255, 1), knob_color=(219, 0, 255, 1),
        knob_border_color=(255, 255, 255, 1), track_corner_radius=2, knob_corner_radius=11, knob_width=20,
        knob_height=20, knob_size=5, show_text=False, value_text=0,
        min_text=0, max_text=100, font='assets/fonts/Poppins/Poppins.ttf', font_size=12,
        font_color=(0, 0, 0, 1), bold=False, italic=False, underline=False,
        strikethrough=False, opacity=1, track_border_thickness=0, knob_border_thickness=3,
        track_height=10, is_visible=True, is_disabled=False)

    ui_page["image"] = pv.PvOpencvImage(container=window, x=938, y=77, width=352,
        height=273, idle_color=(217, 217, 217, 1), scale=1, corner_radius=10,
        flip_v=False, flip_h=False, rotate=0, border_color=(0, 0, 0, 1),
        border_hover_color=None, border_thickness=0, border_style="solid", is_visible=True,
        fill=True, opacity=1, on_hover=None, on_click=None,
        on_release=None, tag=None)

    return ui_page
