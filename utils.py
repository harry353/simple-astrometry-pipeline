import os
import numpy as np
from scipy.signal import windows, wiener
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt


def save_phase_correlation_map(img1, img2, haystack_path, output_path=None):
    """Compute and save the normalised cross-power spectrum (phase correlation map).

    Normalising the cross-power spectrum to unit magnitude before the IFFT
    means every frequency contributes equally regardless of its energy,
    producing a sharp impulse at the true shift rather than a broad peak.
    """

    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    cross = F1 * np.conj(F2)
    denom = np.abs(cross)
    denom[denom == 0] = 1e-10
    corr_map = np.fft.fftshift(np.fft.ifft2(cross / denom).real)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(haystack_path),
            os.path.splitext(os.path.basename(haystack_path).replace("haystack", "correlation_map"))[0] + ".png"
        )
    plt.imsave(output_path, corr_map, cmap='magma', vmin=corr_map.min(), vmax=corr_map.max())
    print(f"Saved correlation map to {output_path}")


def prepare_images(haystack, needle):
    H, W = haystack.shape

    with np.errstate(divide='ignore', invalid='ignore'):
        haystack = np.nan_to_num(wiener(haystack, mysize=7),  nan=0.0)
        needle   = np.nan_to_num(wiener(needle,   mysize=11), nan=0.0)

    canvas = pad_to_size(needle, H, W)

    window     = np.outer(windows.hann(H), windows.hann(W))
    w_haystack = np.copy(haystack)
    w_canvas   = np.copy(canvas)
    return w_haystack, w_canvas


def pad_to_size(img, H, W):
    canvas = np.zeros((H, W), dtype=np.float64)
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
