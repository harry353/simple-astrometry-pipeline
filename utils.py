import os
import numpy as np
from scipy.signal import windows, wiener
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from skimage.feature import match_template


def match_template_subpixel(haystack, needle):
    import constants
    r       = constants.SUBPIXEL_AREA // 2
    result  = match_template(haystack, needle, pad_input=True)
    yi, xi  = np.unravel_index(np.argmax(result), result.shape)

    patch = result[yi-r:yi+r+1, xi-r:xi+r+1]

    # Fit a parabola f(x) = a*x^2 + b*x + c through the three centre values
    # along each axis (at x = -1, 0, +1). The peak is at x = -b / (2a).
    for axis, (f_neg, f_0, f_pos) in enumerate([
        (patch[r-1, r], patch[r, r], patch[r+1, r]),   # y axis
        (patch[r, r-1], patch[r, r], patch[r, r+1]),   # x axis
    ]):
        a = (f_pos + f_neg - 2*f_0) / 2   # curvature
        b = (f_pos - f_neg)          / 2   # slope at centre
        offset = -b / (2 * a)              # peak offset from centre pixel
        if axis == 0:
            dy = offset
        else:
            dx = offset

    # skimage pad_input=True pads by nh//2 rows and nw//2 cols, making its
    # "centre" pixel index nh//2 (0-based). The geometric centre is (nh-1)/2.
    # For even template sizes these differ by 0.5 px — correct for that here.
    nh, nw  = needle.shape
    x_bias  = nw // 2 - (nw - 1) / 2   # 0.5 for even, 0.0 for odd
    y_bias  = nh // 2 - (nh - 1) / 2

    H, W    = haystack.shape
    shift_x = (xi + dx - x_bias) - W / 2
    shift_y = (yi + dy - y_bias) - H / 2
    return shift_x, shift_y, result


def save_correlation_map(corr_map, haystack_path):
    pair_num    = os.path.basename(os.path.dirname(haystack_path))
    output_path = os.path.join(os.path.dirname(haystack_path), f"correlation_map_{pair_num}.png")
    plt.imsave(output_path, corr_map, cmap='magma')
    print(f"Saved correlation map to {output_path}")


def prepare_images(haystack, needle):
    H, W = haystack.shape

    with np.errstate(divide='ignore', invalid='ignore'):
        haystack = np.nan_to_num(wiener(haystack, mysize=7),  nan=0.0)
        needle   = np.nan_to_num(wiener(needle,   mysize=11), nan=0.0)

    padded_needle = pad_to_size(needle, H, W)

    window     = np.outer(windows.hann(H), windows.hann(W))
    w_haystack = np.copy(haystack)
    w_padded_needle   = np.copy(padded_needle)
    return w_haystack, w_padded_needle


def pad_to_size(img, H, W):
    canvas = np.full((H, W), img.mean(), dtype=np.float64)
    nh, nw = img.shape
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    return canvas


def load_fits(haystack_path, needle_path):
    try:
        with fits.open(haystack_path) as hdul:
            haystack = hdul[0].data
            header_haystack = hdul[0].header

        with fits.open(needle_path) as hdul:
            needle = hdul[0].data
            header_needle = hdul[0].header
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure generation scripts have run.")
        return None, None, None, None

    print(f"Loaded Haystack: {haystack.shape}, Needle: {needle.shape}")
    return haystack, header_haystack, needle, header_needle


def print_localisation_accuracy(shift_x, shift_y, header_needle, header_haystack):
    if header_needle is None or 'CRVAL1' not in header_needle:
        return

    px_scale_arcsec = abs(header_haystack.get('CDELT2', 0)) * 3600

    H, W = header_haystack['NAXIS2'], header_haystack['NAXIS1']
    det_cx = W // 2 + shift_x
    det_cy = H // 2 + shift_y

    wcs_h = WCS(header_haystack)
    true_cx, true_cy = wcs_h.world_to_pixel_values(header_needle['CRVAL1'], header_needle['CRVAL2'])

    dx_true = det_cx - true_cx
    dy_true = det_cy - true_cy
    dist_true = np.sqrt(dx_true ** 2 + dy_true ** 2)

    def row(label, px=None, cx=None, cy=None, scalar=None):
        if cx is not None:
            pos = f"({cx:.2f}, {cy:.2f})"
            print(f"  {label:<30} {pos:<22}")
        elif scalar is not None:
            print(f"  {label:<30} {scalar}")
        else:
            arcsec = px * px_scale_arcsec if px_scale_arcsec else None
            px_str = f"{px:.4f} px"
            as_str = f"{arcsec:.4f} arcsec" if arcsec is not None else ""
            print(f"  {label:<30} {px_str:<16} {as_str}")

    print(f"  {'':-<62}")
    print(f"  {'':<30} {'px':<16} {'arcsec'}")
    print(f"  {'':-<62}")

    if 'WERR_XPX' in header_needle:
        wcs_err_x = header_needle['WERR_XPX']
        wcs_err_y = header_needle['WERR_YPX']
        ref_cx = true_cx + wcs_err_x
        ref_cy = true_cy + wcs_err_y
        wcs_err_mag = np.sqrt(wcs_err_x ** 2 + wcs_err_y ** 2)
        row("True centre (cx, cy)",          cx=true_cx,  cy=true_cy)
        row("WCS-predicted centre (cx, cy)", cx=ref_cx,   cy=ref_cy)
        row("WCS error magnitude",           px=wcs_err_mag)
        row("  dx",                          px=abs(wcs_err_x))
        row("  dy",                          px=abs(wcs_err_y))

    det_ra, det_dec = wcs_h.pixel_to_world_values(det_cx, det_cy)

    print(f"  {'':-<62}")
    row("Detected centre (cx, cy)",      cx=det_cx,   cy=det_cy)
    row("Detected centre (RA, Dec)",     scalar=f"{det_ra:.6f} deg,  {det_dec:.6f} deg")
    row("Detection error",               px=dist_true)
    row("  dx",                          px=abs(dx_true))
    row("  dy",                          px=abs(dy_true))
    print(f"  {'':-<62}")
