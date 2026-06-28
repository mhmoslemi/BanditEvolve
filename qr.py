"""
Bad-ass styled QR codes for a poster.
- Rounded "pill" modules + rounded finder eyes.
- Radial gradient color (per-job).
- Optional logo in the middle (on a clean white plate).
- Error correction H (~30%) so the center logo doesn't break scanning.

Usage:
    python3 qr.py
Edit the JOBS dict: name -> (url, logo_path_or_None, (center_color, edge_color)).
"""
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask
from PIL import Image, ImageDraw

# ---- config -----------------------------------------------------------------
BOX = 60            # pixels per QR module -> bigger = higher res
BORDER = 2          # quiet zone in modules (>=2 keeps it scannable)
LOGO_FRAC = 0.350    # logo width as fraction of QR width (keep <= 0.30 for level H)
PAD_FRAC = 0.10     # white padding around the logo, as fraction of logo size

BACK_COLOR = (255, 255, 255)  # background

# radial gradient color schemes: (center_color, edge_color)
BLUE  = ((86, 199, 224), (19, 78, 150))   # light teal -> deep blue
BLACK = ((110, 110, 110), (0, 0, 0))      # light gray -> black

JOBS = {
    # name : (url, logo_path or None, (center_color, edge_color))
    # "qr_github":   ("https://github.com/mhmoslemi/COBRA",
    #                 "github-logo.png", BLACK),
    "qr_linkedin": ("https://www.linkedin.com/in/mohammad-hosein-moslemi/",
                    "linkedin-logo.png", BLUE),
}


def make_qr(url, out_path, logo_path=None, colors=BLUE):
    center_color, edge_color = colors
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, box_size=BOX, border=BORDER)
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(radius_ratio=1),
        eye_drawer=RoundedModuleDrawer(radius_ratio=0.5),
        color_mask=RadialGradiantColorMask(
            back_color=BACK_COLOR, center_color=center_color, edge_color=edge_color
        ),
    ).convert("RGBA")
    W, H = img.size

    if logo_path:
        try:
            logo = Image.open(logo_path).convert("RGBA")
        except FileNotFoundError:
            logo = None
        if logo is not None:
            target = int(W * LOGO_FRAC)
            scale = target / max(logo.size)          # fit longest side, can upscale
            logo = logo.resize((max(1, int(logo.size[0] * scale)),
                                max(1, int(logo.size[1] * scale))), Image.LANCZOS)
            lw, lh = logo.size
            pad = int(max(lw, lh) * PAD_FRAC)
            plate = Image.new("RGBA", (lw + 2 * pad, lh + 2 * pad), (0, 0, 0, 0))
            d = ImageDraw.Draw(plate)
            d.rounded_rectangle([0, 0, plate.size[0] - 1, plate.size[1] - 1],
                                radius=int(pad * 1.6) + 8, fill="white")
            plate.alpha_composite(logo, (pad, pad))
            img.alpha_composite(plate, ((W - plate.size[0]) // 2,
                                        (H - plate.size[1]) // 2))

    img.convert("RGB").save(out_path, "PNG")
    print(f"wrote {out_path}  ({img.size[0]}x{img.size[1]} px)")


if __name__ == "__main__":
    for name, (url, logo, colors) in JOBS.items():
        make_qr(url, f"{name}.png", logo_path=logo, colors=colors)
