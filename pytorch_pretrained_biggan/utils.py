# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from io import BytesIO
import logging

import numpy as np
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)

def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def convert_to_images(obj):
    try:
        import PIL
    except ImportError:
        raise ImportError("Please install Pillow to use images: pip install Pillow")

    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()

    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

def save_as_images(obj, file_name='output'):
    img = convert_to_images(obj)

    for i, out in enumerate(img):
        current_file_name = file_name + '_%d.png' % i
        logger.info("Saving image to {}".format(current_file_name))
        out.save(current_file_name, 'png')

def display_in_terminal(obj):
    try:
        from libsixel import (sixel_output_new, sixel_dither_new, sixel_dither_initialize,
                              sixel_dither_set_palette, sixel_dither_set_pixelformat,
                              sixel_dither_get, sixel_encode, sixel_dither_unref,
                              sixel_output_unref, SIXEL_PIXELFORMAT_RGBA8888,
                              SIXEL_PIXELFORMAT_RGB888, SIXEL_PIXELFORMAT_PAL8,
                              SIXEL_PIXELFORMAT_G8, SIXEL_PIXELFORMAT_G1)
    except ImportError:
        raise ImportError("Display in Terminal requires libsixel"
                          "and a libsixel compatible terminal."
                          "Please read info at https://github.com/saitoha/libsixel"
                          "and install with pip install python-libsixel")

    s = BytesIO()

    images = convert_to_images(obj)
    widths, heights = zip(*(i.size for i in images))

    output_width = sum(widths)
    output_height = max(heights)

    output_image = PIL.Image.new('RGB', (output_width, output_height))

    x_offset = 0
    for im in images:
        output_image.paste(im, (x_offset,0))
        x_offset += im.size[0]

    try:
        data = image.tobytes()
    except NotImplementedError:
        data = image.tostring()
    output = sixel_output_new(lambda data, s: s.write(data), s)

    try:
        if image.mode == 'RGBA':
            dither = sixel_dither_new(256)
            sixel_dither_initialize(dither, data, output_width, output_height, SIXEL_PIXELFORMAT_RGBA8888)
        elif image.mode == 'RGB':
            dither = sixel_dither_new(256)
            sixel_dither_initialize(dither, data, output_width, output_height, SIXEL_PIXELFORMAT_RGB888)
        elif image.mode == 'P':
            palette = image.getpalette()
            dither = sixel_dither_new(256)
            sixel_dither_set_palette(dither, palette)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_PAL8)
        elif image.mode == 'L':
            dither = sixel_dither_get(SIXEL_BUILTIN_G8)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_G8)
        elif image.mode == '1':
            dither = sixel_dither_get(SIXEL_BUILTIN_G1)
            sixel_dither_set_pixelformat(dither, SIXEL_PIXELFORMAT_G1)
        else:
            raise RuntimeError('unexpected image mode')
        try:
            sixel_encode(data, output_width, output_height, 1, dither, output)
            print(s.getvalue().decode('ascii'))
        finally:
            sixel_dither_unref(dither)
    finally:
        sixel_output_unref(output)
