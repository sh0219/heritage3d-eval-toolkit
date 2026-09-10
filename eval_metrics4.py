#!/usr/bin/env python3
"""
Image Quality Evaluation Script for NeRF / 3D Reconstruction
============================================================

Evaluates PSNR, SSIM, and LPIPS between a set of rendered images
and corresponding ground truth images.

Changes in eval_metrics4 (relative to eval_metrics3):

  ICC handling fixes (the main reason for this revision):
    - eval_metrics3 classified the ICC profile by scanning the raw profile
      bytes with latin-1 decoding.  Real ICC profiles store the human-readable
      description tag as UTF-16BE ("mluc" type), so the latin-1 substring
      search never matched and every profile (including genuine sRGB ones) was
      classified as "generic RGB (unknown)".  eval_metrics4 extracts the
      description with Pillow's public ``ImageCms.getProfileDescription()``
      API and matches keywords against a normalised description.
    - eval_metrics3 built the source profile with private Pillow C-API calls
      (``ImageCms.core.buildProfileFromOpenProfiles`` /
      ``profileFromData``) that do not exist on Pillow >= 8, so every ICC->sRGB
      conversion silently fell back to a plain ``convert("RGB")`` — the colour
      transform was never actually applied.  eval_metrics4 constructs the
      source profile from an in-memory ``io.BytesIO`` wrapper, which is the
      documented public API, so the conversion genuinely runs.

  Robustness fixes:
    - PSNR of two identical images is no longer ``inf``; the MSE is floored at
      1e-10 so the result is a finite, sensible number (100.0 dB).
    - After bicubic resizing of the render, the tensor is re-clamped to
      [0, 1] to remove the small overshoot that bicubic interpolation can
      introduce (which would otherwise leak into PSNR/SSIM/LPIPS).
    - Non-ASCII console output (the "⚠" warning glyph) is emitted through
      an error-tolerant stdout wrapper so the script no longer crashes with
      UnicodeEncodeError on consoles that cannot encode it (e.g. Chinese
      Windows / GBK).
    - Removed the unused ``os`` import and a dead no-op block that was left in
      the image-mode conversion logic.

Resize strategy (unchanged from eval_metrics3):
    - eval_metrics2 centre-cropped the larger image to match the smaller one.
      This destroys camera-geometry correspondence (changes FOV & principal
      point) and is incorrect for NeRF / 3DGS / MVS evaluation.
    - eval_metrics4 resizes render to GT dimensions via bicubic interpolation
      (F.interpolate).  GT is never modified — it represents the real
      observation and defines the reference coordinate frame.

Colour-space check (unchanged from eval_metrics3):
    - Inspects the **ICC profile** embedded in the image and classifies it as
      sRGB, Adobe RGB, Display P3, or generic/unknown, printing a warning when
      the profile is non-sRGB or absent.
    - Non-sRGB images are **actually converted to sRGB** via PIL.ImageCms
      (requires the ``littlecms2`` system library; falls back to a warning if
      unavailable).

Image-meta summary (unchanged from eval_metrics3):
    - A per-pair summary is printed before metrics: image mode, ICC profile
      classification, and original dimensions, so the experimenter can audit
      the dataset at a glance.

Implementation references:
  - PSNR: Instant-NGP method  (NVlabs/instant-ngp, scripts/common.py)
          MSE averaged over all RGB channels, then PSNR = -10 * log10(MSE).
          Images are clipped to [0, 1] before computation.
  - SSIM: 3D Gaussian Splatting method (graphdeco-inria/gaussian-splatting,
          utils/loss_utils.py). 11x11 Gaussian window (sigma=1.5), RGB three-
          channel group-conv, global pixel+channel mean as the SSIM score.
  - LPIPS: Official richzhang/PerceptualSimilarity package (pip install lpips).
          Inputs are normalized to [-1, 1] before being passed to the AlexNet-
          based LPIPS model, conforming to the official usage spec.

Requirements:
  Python 3.8.3  |  PyTorch 1.7.1  |  CUDA 11.7
  pip install lpips torch torchvision
  Optional: system library ``libltdl7`` / ``littlecms2`` for ICC->sRGB conversion.

Usage:
  python eval_metrics4.py --gt_dir /path/to/groundtruth --render_dir /path/to/renders
  python eval_metrics4.py --gt_dir ./gt --render_dir ./renders --device cuda
  python eval_metrics4.py --gt_dir ./gt --render_dir ./renders --save_per_image results.csv
  python eval_metrics4.py --gt_dir ./gt --render_dir ./renders --no_icc_convert
"""

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0.  Console-output safety: never crash on non-encodable characters.
#     (e.g. the "⚠" warning glyph on a GBK / cp936 Windows console)
# ---------------------------------------------------------------------------
class _SafeStdout:
    """Write-through wrapper that replaces characters the console cannot encode."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, s: str) -> int:
        try:
            return self._stream.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._stream, "encoding", None) or "ascii"
            return self._stream.write(s.encode(enc, errors="replace").decode(enc))

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


if hasattr(sys.stdout, "encoding") and sys.stdout.encoding is not None:
    try:
        "⚠".encode(sys.stdout.encoding)
    except (UnicodeEncodeError, LookupError):
        sys.stdout = _SafeStdout(sys.stdout)

# ---------------------------------------------------------------------------
# 0b.  Minimal version check – soft warning only
# ---------------------------------------------------------------------------
_PY_MAJOR, _PY_MINOR = sys.version_info[:2]
if (_PY_MAJOR, _PY_MINOR) < (3, 8):
    print("Warning: Python >= 3.8 is recommended (current: {}.{})".format(
        _PY_MAJOR, _PY_MINOR))

_TORCH_VER = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _TORCH_VER < (1, 7):
    print("Warning: PyTorch >= 1.7.1 is recommended (current: {})".format(
        torch.__version__))


# ---------------------------------------------------------------------------
# 1.  PSNR  
# ---------------------------------------------------------------------------
def psnr_instant_ngp(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Compute PSNR following the Instant-NGP convention.

    PSNR = -10 * log10( MSE )

    where MSE is the mean squared error over all RGB channels and all pixels.
    Both inputs are clipped to [0, 1] beforehand, matching the behaviour in
    NVlabs/instant-ngp/scripts/run.py and scripts/common.py.

    To keep the value finite for two identical images (MSE = 0), the MSE is
    floored at 1e-10 (-> PSNR = 100.0 dB) rather than returning ``inf``.

    Parameters
    ----------
    img : torch.Tensor  shape (C, H, W)  or  (1, C, H, W)
    ref : torch.Tensor  same shape as img

    Returns
    -------
    psnr_val : scalar torch.Tensor  (dB)
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)   # (1, C, H, W)
        ref = ref.unsqueeze(0)

    # Instant-NGP clips to [0, 1] and applies sRGB before computing MSE.
    # For standard PNG/JPG inputs we are already in sRGB, so clipping suffices.
    img = torch.clamp(img, 0.0, 1.0)
    ref = torch.clamp(ref, 0.0, 1.0)

    mse = torch.mean((img - ref) ** 2)
    # Floor MSE so log10 never receives zero (identical-image edge case).
    mse = torch.clamp(mse, min=1e-10)
    # mse2psnr:  -10.0 * log10(mse)
    psnr_val = -10.0 * torch.log10(mse)
    return psnr_val


# ---------------------------------------------------------------------------
# 2.  SSIM  
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Discrete 1-D Gaussian window (mirrors gaussian() in loss_utils.py)."""
    coords = torch.arange(size, dtype=torch.float32)
    centre = size // 2
    gauss = torch.exp(-((coords - centre) ** 2) / (2.0 * sigma ** 2))
    return gauss / gauss.sum()


def _create_window_2d(window_size: int, channel: int) -> torch.Tensor:
    """
    Build a 2-D Gaussian window [channel, 1, window_size, window_size].

    Identical to create_window() in graphdeco-inria/gaussian-splatting
    utils/loss_utils.py.
    """
    _1d = _gaussian_kernel_1d(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Global constants mirroring those in the 3DGS repository.
_C1_SSIM = 0.01 ** 2   # (K1 * L)^2  with L=1, K1=0.01
_C2_SSIM = 0.03 ** 2   # (K2 * L)^2  with L=1, K2=0.03
_SSIM_WINDOW_SIZE = 11


def ssim_3dgs(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM following the 3D Gaussian Splatting convention.

    Key characteristics (see utils/loss_utils.py in the 3DGS repo):
      - 11x11 2-D Gaussian kernel, sigma = 1.5
      - RGB three-channel processing (no luminance conversion) via group conv2d
      - Stability constants C1 = 0.01^2, C2 = 0.03^2  (L = 1.0)
      - Score = global mean over all channels and all pixels of the SSIM map

    Parameters
    ----------
    img : torch.Tensor  shape (C, H, W)  or  (1, C, H, W)
    ref : torch.Tensor  same shape as img

    Returns
    -------
    ssim_val : scalar torch.Tensor
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)   # (1, C, H, W)
        ref = ref.unsqueeze(0)

    channel = img.size(1)
    window = _create_window_2d(_SSIM_WINDOW_SIZE, channel)

    # Move window to the same device / dtype as input
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    # ---- local statistics via grouped 2-D convolution ----
    mu1 = F.conv2d(img, window, padding=_SSIM_WINDOW_SIZE // 2, groups=channel)
    mu2 = F.conv2d(ref, window, padding=_SSIM_WINDOW_SIZE // 2, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img * img, window,
                         padding=_SSIM_WINDOW_SIZE // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(ref * ref, window,
                         padding=_SSIM_WINDOW_SIZE // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img * ref, window,
                         padding=_SSIM_WINDOW_SIZE // 2, groups=channel) - mu1_mu2

    # ---- SSIM map ----
    numerator = (2.0 * mu1_mu2 + _C1_SSIM) * (2.0 * sigma12 + _C2_SSIM)
    denominator = (mu1_sq + mu2_sq + _C1_SSIM) * (sigma1_sq + sigma2_sq + _C2_SSIM)
    ssim_map = numerator / denominator

    # Global mean over batch, channels, height, width (3DGS default behaviour)
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# 3.  LPIPS  –  Official richzhang/PerceptualSimilarity package
# ---------------------------------------------------------------------------
def _build_lpips(device: str):
    """
    Create the official LPIPS evaluator (AlexNet backbone).

    Unlike the 3DGS-bundled lpipsPyTorch (which has a known normalisation
    discrepancy – applying [-1, 1]-tuned z-score stats to [0, 1] images),
    this uses the canonical ``lpips`` pip package with proper semantics:
    input image tensors MUST be in the [-1, 1] range.
    """
    try:
        import lpips as lpips_mod
    except ImportError:
        print("\nERROR: The 'lpips' package is required for LPIPS computation.")
        print("       Install it with:  pip install lpips\n")
        sys.exit(1)

    loss_fn = lpips_mod.LPIPS(net="alex", version="0.1").to(device)
    loss_fn.eval()
    return loss_fn


def lpips_official(img: torch.Tensor, ref: torch.Tensor,
                   loss_fn) -> float:
    """
    Compute LPIPS between two images using the official richzhang/PerceptualSimilarity
    implementation.

    The function converts [0, 1]-range inputs to [-1, 1] (as required by the
    official LPIPS model) before calling the forward pass.

    Parameters
    ----------
    img : torch.Tensor  shape (1, 3, H, W)
    ref : torch.Tensor  shape (1, 3, H, W)
    loss_fn : lpips.LPIPS instance

    Returns
    -------
    lpips_val : float
    """
    if img.size(1) != 3:
        raise ValueError("LPIPS expects 3-channel RGB input, got {} channels".format(
            img.size(1)))

    # Official LPIPS requires inputs in [-1, 1].
    img_norm = img * 2.0 - 1.0
    ref_norm = ref * 2.0 - 1.0

    with torch.no_grad():
        dist = loss_fn.forward(img_norm, ref_norm)
    # dist shape: (1, 1, 1, 1) for non-spatial mode
    return float(dist.item())


# ---------------------------------------------------------------------------
# 4.  ICC profile utilities  (FIXED in eval_metrics4)
# ---------------------------------------------------------------------------
# Known ICC profile description substrings (matched case-insensitively against
# the *decoded* description tag text returned by Pillow's public
# ``ImageCms.getProfileDescription()`` API).
_ICC_DESCRIPTIONS = {
    "srgb": "sRGB",
    "srgb_iec61966": "sRGB IEC61966-2.1",
    "adobergb": "Adobe RGB (1998)",
    "display_p3": "Display P3",
    "display p3": "Display P3",
    "p3": "Display P3",
    "prophoto": "ProPhoto RGB",
    "cmyk": "CMYK",
    "swop": "SWOP (CMYK)",
    "coated fogra": "FOGRA (CMYK)",
}


def _normalise_icc_text(text: str) -> str:
    """Lower-case and strip everything except ASCII letters/digits so keyword
    matching tolerates whitespace, punctuation and trailing newlines that
    commonly appear in ICC description tags (e.g. "sRGB built-in\\n")."""
    return "".join(ch for ch in text.lower() if ch.isalnum() and ch.isascii())


def _classify_icc_profile(icc_bytes: Optional[bytes]) -> str:
    """
    Classify an ICC profile from its raw byte stream.

    Unlike eval_metrics3 (which scanned the raw bytes with latin-1 decoding
    and therefore failed to match the UTF-16BE "mluc" description tag used by
    real profiles), this extracts the description through Pillow's public
    ``ImageCms.getProfileDescription()`` API and matches normalised keywords.

    Parameters
    ----------
    icc_bytes : bytes or None
        The raw ICC profile as returned by ``im.info.get("icc_profile")``.

    Returns
    -------
    label : str
        One of: "sRGB", "Adobe RGB", "Display P3", "ProPhoto RGB",
        "generic CMYK", "generic RGB (unknown)", or "no ICC profile".
    """
    if icc_bytes is None or len(icc_bytes) == 0:
        return "no ICC profile"

    # Quick sanity check: the first four bytes of every ICC profile are the
    # profile size as a big-endian uint32.  If they don't look plausible the
    # bytes may be an Exif colour-space tag rather than a real ICC profile.
    _size_tag = int.from_bytes(icc_bytes[0:4], "big")
    if _size_tag < 128 or _size_tag > 50_000_000:
        return "no ICC profile (malformed header)"

    # 1) Try Pillow's public description extractor first.
    try:
        from PIL import ImageCms
        desc = ImageCms.getProfileDescription(io.BytesIO(icc_bytes)) or ""
    except Exception:
        desc = ""

    if desc.strip():
        norm = _normalise_icc_text(desc)
        for keyword, label in _ICC_DESCRIPTIONS.items():
            if _normalise_icc_text(keyword) in norm:
                return label

    # 2) Fallback: search the raw bytes (both latin-1 and UTF-16BE views).
    latin_text = icc_bytes.decode("latin-1", errors="ignore").lower()
    utf16_text = icc_bytes.decode("utf-16-be", errors="ignore").lower()
    for keyword, label in _ICC_DESCRIPTIONS.items():
        if keyword in latin_text or keyword in utf16_text:
            return label

    # 3) Final fallback: check the device class (byte offset 12–15).
    if len(icc_bytes) >= 16:
        dev_class = icc_bytes[12:16].decode("ascii", errors="ignore")
        if dev_class in ("mntr", "scnr", "spac"):
            return "generic RGB (unknown)"
        elif dev_class == "prtr":
            return "generic CMYK"

    return "generic (unknown)"


def _try_convert_to_srgb(path: str) -> Tuple[Optional[str], object]:
    """
    Open an image, inspect its ICC profile, and attempt to convert it to sRGB
    using PIL.ImageCms if the profile is non-sRGB.

    Parameters
    ----------
    path : str  file path to the image

    Returns
    -------
    (icc_label, pil_image) : (str or None, PIL.Image.Image)
        - icc_label: human-readable ICC classification (or None on error).
        - pil_image: the (possibly converted) PIL Image in RGB mode, or
          the original image if conversion was not possible.
    """
    from PIL import Image

    icc_label = None
    try:
        im = Image.open(path)
    except Exception:
        return None, None

    # ---- inspect ICC profile ----
    icc_bytes = im.info.get("icc_profile", None)
    icc_label = _classify_icc_profile(icc_bytes)

    # If no ICC profile or already sRGB, just ensure RGB mode.
    if icc_label in ("sRGB", "sRGB IEC61966-2.1", "no ICC profile",
                     "no ICC profile (malformed header)"):
        if im.mode != "RGB":
            im = im.convert("RGB")
        return icc_label, im

    # ---- non-sRGB profile detected; attempt ICC->sRGB conversion ----
    try:
        from PIL import ImageCms
        srgb_profile = ImageCms.createProfile("sRGB")
        if icc_bytes is not None and len(icc_bytes) > 0:
            # Public Pillow API: ImageCmsProfile accepts a file-like object
            # (eval_metrics3 used private C-API calls that are gone on Pillow >= 8).
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        else:
            # No source profile to convert from.
            if im.mode != "RGB":
                im = im.convert("RGB")
            return icc_label, im

        # Build the transform.  If the image is CMYK we need to go
        # CMYK->RGB; otherwise use RGB->RGB (gamut mapping only).
        if im.mode == "CMYK":
            intent = ImageCms.Intent.PERCEPTUAL
            transform = ImageCms.buildTransform(
                src_profile, srgb_profile, "CMYK", "RGB",
                intent, flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
        else:
            # Ensure RGB mode before colour-space conversion.
            if im.mode != "RGB":
                im = im.convert("RGB")
            intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
            transform = ImageCms.buildTransform(
                src_profile, srgb_profile, "RGB", "RGB",
                intent, flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)

        im = ImageCms.applyTransform(im, transform)
        return icc_label, im

    except ImportError:
        # ImageCms not available — likely missing littlecms2 system library.
        # Fall back to naive .convert("RGB") and note the failure.
        if im.mode != "RGB":
            im = im.convert("RGB")
        return icc_label + " (no ImageCms)", im
    except Exception:
        # Any unexpected error during ICC conversion — fall back.
        if im.mode != "RGB":
            im = im.convert("RGB")
        return icc_label + " (convert failed)", im


# ---------------------------------------------------------------------------
# 5.  Image loading  (modified in eval_metrics4)
# ---------------------------------------------------------------------------
def _read_image_as_tensor(path: str, device: str,
                          do_icc_convert: bool = True) -> torch.Tensor:
    """
    Read a single image from disk and return as a (1, C, H, W) float32
    tensor on the given device, pixel range [0, 1].

    When *do_icc_convert* is True, non-sRGB ICC profiles are converted to
    sRGB via PIL.ImageCms before the image is turned into a tensor.

    Parameters
    ----------
    path : str
    device : str
    do_icc_convert : bool

    Returns
    -------
    tensor : torch.Tensor  shape (1, 3, H, W)
    """
    try:
        from torchvision import transforms
    except ImportError:
        print("\nERROR: torchvision is required. Install with: pip install torchvision\n")
        sys.exit(1)

    from PIL import Image, UnidentifiedImageError

    try:
        if do_icc_convert:
            # Use ICC-aware loading path.
            _, pil_img = _try_convert_to_srgb(path)
            if pil_img is None:
                raise IOError("Could not open image '{}'".format(path))
            # _try_convert_to_srgb already returns an RGB image.
        else:
            pil_img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise IOError("Could not open image '{}': {}".format(path, e))

    to_tensor = transforms.ToTensor()   # [0, 1] float32,  (C, H, W)
    tensor = to_tensor(pil_img).unsqueeze(0).to(device)  # (1, C, H, W)
    return tensor


# ---------------------------------------------------------------------------
# 6.  Dimension consistency – render → GT resize  (CHANGED in eval_metrics3)
# ---------------------------------------------------------------------------
def _resize_render_to_gt(
    gt_tensor: torch.Tensor,
    render_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Ensure the render tensor has the same spatial dimensions as the GT tensor.

    If sizes differ, the **render is bicubicly resized to match GT**.
    GT is never modified — it represents the real observation and defines the
    reference coordinate frame.  This preserves the camera-geometry
    relationship (FOV, principal point) that centre-cropping would destroy.

    After resizing, the render is re-clamped to [0, 1] to remove the small
    overshoot bicubic interpolation can produce (which would otherwise leak
    into the metrics).

    Parameters
    ----------
    gt_tensor : torch.Tensor      shape (1, C, H_gt, W_gt)
    render_tensor : torch.Tensor  shape (1, C, H_rn, W_rn)

    Returns
    -------
    gt_tensor, render_tensor : torch.Tensor
        Tensors with matching (1, C, H, W) shapes.
    was_resized : bool
        True if render was resized.
    """
    _, _, h_gt, w_gt = gt_tensor.shape
    _, _, h_rn, w_rn = render_tensor.shape

    if h_gt == h_rn and w_gt == w_rn:
        return gt_tensor, render_tensor, False

    print("  ⚠  SIZE MISMATCH:  GT ({}, {})  vs  Render ({}, {}): "
          "resizing render → GT via bicubic interpolation.".format(
              h_gt, w_gt, h_rn, w_rn))

    render_tensor = F.interpolate(
        render_tensor,
        size=(h_gt, w_gt),
        mode="bicubic",
        align_corners=False,
    )
    # Remove bicubic overshoot that can push values outside [0, 1].
    render_tensor = torch.clamp(render_tensor, 0.0, 1.0)
    return gt_tensor, render_tensor, True


# ---------------------------------------------------------------------------
# 7.  Image metadata inspection  (NEW in eval_metrics3)
# ---------------------------------------------------------------------------
def _inspect_image_meta(path: str, name: str,
                        do_icc_convert: bool) -> Tuple[str, str, int, int]:
    """
    Inspect an image and return its (mode_label, icc_label, height, width).

    This opens the image twice (once for ICC, once for dimension) which is
    slightly wasteful but keeps the loader fast-path clean.  The ICC check
    reuses ``_try_convert_to_srgb`` to avoid duplicating logic.

    Parameters
    ----------
    path : str
    name : str             human label ("GT" or "Render")
    do_icc_convert : bool  whether ICC->sRGB conversion will be attempted

    Returns
    -------
    mode_label : str   e.g. "RGB" or "RGB ← L (greyscale)"
    icc_label : str    e.g. "sRGB", "Adobe RGB", "no ICC profile"
    h, w : int
    """
    from PIL import Image

    mode_label = "?"
    icc_label = "?"
    h = w = 0

    try:
        with Image.open(path) as im:
            native_mode = im.mode
            h, w = im.height, im.width

            mode_label = native_mode
            if native_mode != "RGB":
                mode_label = "RGB ← {} (native)".format(native_mode)

            if do_icc_convert:
                icc_bytes = im.info.get("icc_profile", None)
                icc_label = _classify_icc_profile(icc_bytes)
            else:
                icc_label = "skipped (--no_icc_convert)"
    except Exception:
        mode_label = "error"
        icc_label = "error"

    return mode_label, icc_label, h, w


# ---------------------------------------------------------------------------
# 8.  Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR (Instant-NGP), SSIM (3DGS) and LPIPS (official)"
    )
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground-truth images.")
    parser.add_argument("--render_dir", type=str, required=True,
                        help="Directory containing rendered/predicted images.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu' (default: cuda).")
    parser.add_argument("--save_per_image", type=str, default=None,
                        help="Optional CSV path for per-image metrics.")
    parser.add_argument("--no_icc_convert", action="store_true",
                        help="Skip ICC->sRGB conversion (still inspect profiles).")
    args = parser.parse_args()

    do_icc_convert = not args.no_icc_convert

    # ------------------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------
    # Collect matching image pairs
    gt_dir = Path(args.gt_dir)
    render_dir = Path(args.render_dir)

    if not gt_dir.is_dir():
        sys.exit("Ground-truth directory not found: {}".format(gt_dir))
    if not render_dir.is_dir():
        sys.exit("Render directory not found: {}".format(render_dir))

    common_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    gt_files = sorted([
        f for f in gt_dir.iterdir()
        if f.suffix.lower() in common_exts and f.is_file()
    ])
    render_files = sorted([
        f for f in render_dir.iterdir()
        if f.suffix.lower() in common_exts and f.is_file()
    ])

    # Match by filename (mutual intersection, so orphan renders are also
    # reported rather than silently ignored).
    render_names = {f.name for f in render_files}
    matched = sorted(f for f in gt_files if f.name in render_names)

    if not matched:
        print("No matching filename pairs found between the two directories.")
        print("  gt dir ({} images): {}".format(len(gt_files), gt_dir))
        print("  render dir ({} images): {}".format(len(render_files), render_dir))
        sys.exit(1)

    if len(gt_files) != len(matched):
        print("Note: {} ground-truth image(s) have no render "
              "counterpart and will be skipped.".format(len(gt_files) - len(matched)))
    if len(render_files) != len(matched):
        print("Note: {} render image(s) have no ground-truth "
              "counterpart and will be skipped.".format(len(render_files) - len(matched)))

    print("\nEvaluating {} image pairs on device '{}' ...".format(len(matched), device))
    if do_icc_convert:
        print("ICC->sRGB conversion: ENABLED")
    else:
        print("ICC->sRGB conversion: DISABLED (--no_icc_convert)")
    print()

    # ------------------------------------------------------------------
    # Initialise LPIPS (AlexNet) – official richzhang package
    lpips_fn = _build_lpips(device)

    per_image_results = []

    # Accumulators for summary stats
    psnr_vals  = []
    ssim_vals  = []
    lpips_vals = []

    # Counters for consistency checks
    resize_count        = 0
    mode_warnings       = 0
    icc_warnings        = 0
    skipped_count       = 0

    # ------------------------------------------------------------------
    # Main loop
    for gtf in matched:
        rtf = render_dir / gtf.name

        # ---- inspect image metadata ----
        gt_mode, gt_icc, gt_h, gt_w = _inspect_image_meta(
            str(gtf), "GT", do_icc_convert)
        rn_mode, rn_icc, rn_h, rn_w = _inspect_image_meta(
            str(rtf), "Render", do_icc_convert)

        # Build a compact per-image header line.
        meta_parts = []
        if gt_mode != "RGB" or rn_mode != "RGB":
            meta_parts.append("mode={}/{}".format(gt_mode, rn_mode))
        # Warn on non-RGB native modes.
        if "←" in gt_mode:
            print("  ⚠  IMAGE MODE: GT native mode is '{}' → RGB".format(
                gt_mode.split("←")[1].strip(" (native)")))
            mode_warnings += 1
        if "←" in rn_mode:
            print("  ⚠  IMAGE MODE: Render native mode is '{}' → RGB".format(
                rn_mode.split("←")[1].strip(" (native)")))
            mode_warnings += 1

        # ICC warnings
        if gt_icc not in ("sRGB", "sRGB IEC61966-2.1", "skipped (--no_icc_convert)"):
            meta_parts.append("GT_ICC={}".format(gt_icc))
            if gt_icc != "no ICC profile":
                icc_warnings += 1
        if rn_icc not in ("sRGB", "sRGB IEC61966-2.1", "skipped (--no_icc_convert)"):
            meta_parts.append("Render_ICC={}".format(rn_icc))
            if rn_icc != "no ICC profile":
                icc_warnings += 1

        if gt_h != rn_h or gt_w != rn_w:
            meta_parts.append("{}x{} vs {}x{}".format(gt_h, gt_w, rn_h, rn_w))

        meta_str = "  [{:<20s}]  ".format(gtf.name[:20])
        if meta_parts:
            meta_str += " | ".join(meta_parts)
        else:
            meta_str += "OK"

        # ---- load images ----
        try:
            gt_tensor     = _read_image_as_tensor(str(gtf), device,
                                                   do_icc_convert)
            render_tensor = _read_image_as_tensor(str(rtf), device,
                                                   do_icc_convert)
        except IOError as e:
            print("{}  SKIP: {}".format(meta_str, e))
            skipped_count += 1
            continue

        # ---- resize render → GT  (CHANGED: was centre-crop, now bicubic) ----
        gt_tensor, render_tensor, was_resized = _resize_render_to_gt(
            gt_tensor, render_tensor)
        if was_resized:
            resize_count += 1

        # -- PSNR  --
        psnr_v = psnr_instant_ngp(render_tensor, gt_tensor).item()

        # -- SSIM  --
        ssim_v = ssim_3dgs(render_tensor, gt_tensor).item()

        # -- LPIPS  --
        lpips_v = lpips_official(render_tensor, gt_tensor, lpips_fn)

        per_image_results.append((gtf.name, psnr_v, ssim_v, lpips_v))
        psnr_vals.append(psnr_v)
        ssim_vals.append(ssim_v)
        lpips_vals.append(lpips_v)

        print("{}  PSNR: {:6.2f}  SSIM: {:.4f}  LPIPS: {:.4f}".format(
            meta_str, psnr_v, ssim_v, lpips_v))

    # ------------------------------------------------------------------
    # Summary
    if not psnr_vals:
        sys.exit("No images were successfully processed.")

    psnr_arr  = np.array(psnr_vals)
    ssim_arr  = np.array(ssim_vals)
    lpips_arr = np.array(lpips_vals)

    print("\n" + "=" * 75)
    print("Summary  –  {} images".format(len(psnr_arr)))
    print("=" * 75)
    print("  PSNR    │  mean: {:7.3f}  min: {:7.3f}  max: {:7.3f}".format(
        psnr_arr.mean(), psnr_arr.min(), psnr_arr.max()))
    print("  SSIM    │  mean: {:7.4f}  min: {:7.4f}  max: {:7.4f}".format(
        ssim_arr.mean(), ssim_arr.min(), ssim_arr.max()))
    print("  LPIPS   │  mean: {:7.4f}  min: {:7.4f}  max: {:7.4f}".format(
        lpips_arr.mean(), lpips_arr.min(), lpips_arr.max()))
    print("=" * 75)

    # ---- consistency-check summary ----
    issues = []
    if resize_count:
        issues.append("Size mismatches (render→GT resized): {}".format(resize_count))
    if mode_warnings:
        issues.append("Non-RGB native mode warnings:       {}".format(mode_warnings))
    if icc_warnings:
        issues.append("Non-sRGB ICC profile warnings:      {}".format(icc_warnings))
    if skipped_count:
        issues.append("Skipped images (I/O errors):        {}".format(skipped_count))

    if issues:
        print("Consistency checks:")
        for line in issues:
            print("  " + line)
        print("=" * 75)
    print()

    # ------------------------------------------------------------------
    # Optional CSV export
    if args.save_per_image:
        csv_path = Path(args.save_per_image)
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "LPIPS"])
            for name, p, s, l in per_image_results:
                writer.writerow([name, "{:.6f}".format(p),
                                 "{:.6f}".format(s), "{:.6f}".format(l)])
        print("Per-image results saved to: {}\n".format(csv_path.resolve()))


if __name__ == "__main__":
    main()
