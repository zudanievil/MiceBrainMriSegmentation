import numpy
import skimage.transform

batch_matmul_einstein = 'bij,bjk->bik'


class RandomHomographyGenerator:
    def __init__(
            self,
            max_rot_z: 'rads' = 3,
            max_scale: 'times-1' = 0.2,
            max_shear: 'side length fraction' = 0.1,
            max_translation: 'side length fraction' = 0.2,
            max_pan_xy: 'not more than 0.001 recommended' = 0.0005
            ):
        self.m = numpy.array([max_rot_z, max_scale, max_scale, max_shear, max_shear,
                              max_translation, max_translation, max_pan_xy, max_pan_xy])

    def __call__(self, height: int, width: int) -> numpy.ndarray:
        rot_z, scale_x, scale_y, shear_x, shear_y, trans_x, trans_y, pan_x, pan_y \
            = numpy.random.uniform(-1, 1, 9) * self.m
        scale_x += 1
        scale_y += 1
        trans_x *= width
        trans_y *= height
        center_x = width / 2.0
        center_y = height / 2.0
        a = numpy.cos(rot_z)
        b = numpy.sin(rot_z)
        return numpy.array(
            [a,  b + shear_x, (scale_x - a) * center_x - b * center_y + trans_x,
             -b, a + shear_y, (scale_y - a) * center_y + b * center_x + trans_y,
             pan_x, pan_y, 1]).reshape(3, 3)


def bbox_to_coords(y0: float, y1: float, x0: float, x1: float, non_cycled=True) -> numpy.ndarray:
    """
    bbox is a representation that is adapted for array slicing
    it goes as [y_low, y_high, x_low, x_high]
    coords are (N, 2) shaped array
    :var non_cycled: if False, last coordinate is a duplication of first (usefull for plotting)
    coordinates have traditional ordering in a pair: (x, y)
    """
    coo = numpy.array([[x0, y1], [x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=numpy.float)
    if non_cycled:
        coo = coo[:-1]
    return coo


def bbox_crop(image, bbox):
    a, b, c, d = bbox
    return image[a:b, c:d]


def rotate_coords(coords: numpy.ndarray, center: 'numpy.ndarray or tuple or int',
                  angle: int, degrees=False) -> numpy.ndarray:
    """
    rotation around Z axis via (2, 2) matrix
    :var coords: assumed shape is (N, 2)
    :var center: int, 2-tuple, numpy.ndarray of shape (N,2)
    :var angle: int.
    :var degrees: if True, angle is converted to radians
    """
    if degrees:
        angle = numpy.pi / 180 * angle
    a = numpy.cos(angle)
    b = numpy.sin(angle)
    m = numpy.array(((a, -b), (b, a)))
    coords = (m @ (coords - center).T).T + center
    return coords


def sigmoid_with_quantiles(img: numpy.ndarray, sigmoid_gain: int = 10, sigmoid_cutoff: float = 0.5,
                           quantiles: (float, float) = (0.07, 0.93)) -> numpy.ndarray:
    flat = img.flatten()
    idx = numpy.argsort(flat)
    low = flat[idx[int(quantiles[0] * len(idx))]]
    high = flat[idx[int(quantiles[1] * len(idx))]]
    img = (img - low) / (high - low)
    img = 1 / (1 + numpy.exp(sigmoid_gain * (sigmoid_cutoff - img)))
    return img


def estimate_affine(src_coo: numpy.ndarray, dst_coo: numpy.ndarray) -> numpy.ndarray:
    """wrap around skimage.transform.AffineTransform"""
    t = skimage.transform.AffineTransform()
    t.estimate(src=dst_coo, dst=src_coo)
    return t.params
