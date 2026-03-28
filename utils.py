import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, wiener
from astropy.io import fits


def denoise(img):
    """Wiener filter to suppress noise before phase correlation.

    Estimates the local noise level from the image itself and adapts
    the filtering per-pixel, so no noise sigma needs to be known.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = wiener(img)
    return np.nan_to_num(result, nan=0.0)


def build_correlation_map(img1, img2):
    """Build the normalised cross-power spectrum correlation map.

    Returns the phase correlation surface with zero-shift centred
    (fftshift applied), suitable for diagnostics and saving.
    """
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    cross = F1 * np.conj(F2)
    denom = np.abs(cross)
    denom[denom == 0] = 1e-10
    return np.fft.fftshift(np.fft.ifft2(cross / denom).real)


def save_correlation(correlation_map, header_haystack, shift_y, shift_x, haystack_path):
    output_path = os.path.join(
        os.path.dirname(haystack_path),
        os.path.basename(haystack_path).replace("haystack", "correlation")
    )
    """Saves the correlation map to a FITS file with peak metadata."""
    H, W = correlation_map.shape
    peak_x = W // 2 + shift_x
    peak_y = H // 2 + shift_y
    y_idx = int(round(np.clip(peak_y, 0, H - 1)))
    x_idx = int(round(np.clip(peak_x, 0, W - 1)))
    hdu = fits.PrimaryHDU(correlation_map, header=header_haystack)
    hdu.header['PEAK_X']   = (peak_x, 'Refined match peak X coordinate')
    hdu.header['PEAK_Y']   = (peak_y, 'Refined match peak Y coordinate')
    hdu.header['PEAK_VAL'] = (correlation_map[y_idx, x_idx], 'Match peak correlation value')
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved correlation map to {output_path}")

    png_path = os.path.splitext(output_path)[0] + ".png"
    plt.imsave(png_path, correlation_map, cmap='magma')
    print(f"Saved correlation map image to {png_path}")


def apply_window(img):
    """Apply a 2-D Hanning window to suppress FFT edge-ringing artifacts."""
    H, W = img.shape
    window = np.outer(windows.hann(H), windows.hann(W))
    return img * window


def prepare_images(haystack, needle):
    """Denoise, pad needle to haystack size, and apply Hanning windows to both.

    Returns (w_haystack, w_canvas) ready to be passed to phase_cross_correlation
    and build_correlation_map.
    """
    H, W = haystack.shape
    w_haystack = apply_window(denoise(haystack))
    w_canvas   = apply_window(pad_to_size(denoise(needle), H, W))
    return w_haystack, w_canvas


def pad_to_size(img, H, W):
    """Zero-pad img to (H, W), centred."""
    canvas = np.zeros((H, W), dtype=np.float64)
    nh, nw = img.shape
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    return canvas


def load_fits(haystack_path, needle_path):
    """Loads haystack and needle FITS files."""
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
    """Prints detected position vs true and reference centres, with distances in px and arcsec."""
    if header_needle is None or 'TRUE_CX' not in header_needle:
        return

    px_scale_arcsec = abs(header_haystack.get('CDELT2', 0)) * 3600

    def fmt(px):
        if px_scale_arcsec:
            return f"{px:.4f} px  ({px * px_scale_arcsec:.4f} arcsec)"
        return f"{px:.4f} px"

    H, W = header_haystack['NAXIS2'], header_haystack['NAXIS1']
    det_cx = W // 2 + shift_x
    det_cy = H // 2 + shift_y

    true_cx = header_needle['TRUE_CX']
    true_cy = header_needle['TRUE_CY']
    dx_true = det_cx - true_cx
    dy_true = det_cy - true_cy
    dist_true = np.sqrt(dx_true ** 2 + dy_true ** 2)
    print(f"True centre          (cx, cy) : ({true_cx:.4f}, {true_cy:.4f})")
    print(f"Detected position    (cx, cy) : ({det_cx:.4f}, {det_cy:.4f})")
    print(f"Distance from true centre     : {fmt(dist_true)}")
    print(f"  dx                          : {fmt(abs(dx_true))}")
    print(f"  dy                          : {fmt(abs(dy_true))}")

    if 'REF_CX' in header_needle:
        ref_cx = header_needle['REF_CX']
        ref_cy = header_needle['REF_CY']
        dx_ref = det_cx - ref_cx
        dy_ref = det_cy - ref_cy
        dist_ref = np.sqrt(dx_ref ** 2 + dy_ref ** 2)
        wcs_err = np.sqrt((ref_cx - true_cx) ** 2 + (ref_cy - true_cy) ** 2)
        print(f"Reference centre     (cx, cy) : ({ref_cx:.4f}, {ref_cy:.4f})")
        print(f"Distance from ref centre      : {fmt(dist_ref)}")
        print(f"  dx                          : {fmt(abs(dx_ref))}")
        print(f"  dy                          : {fmt(abs(dy_ref))}")
        print(f"WCS error magnitude           : {fmt(wcs_err)}")
