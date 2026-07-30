"""EK uygulaması için sade/şık, siyah-beyaz/metalik gri tonlu bir "EK" harf-amblemi üretir.

Tek seferlik, GELİŞTİRME-ZAMANI bir araçtır -- uygulamanın çalışma zamanı hiçbir yerinde
`Pillow` import edilmez/gerekmez (bkz. `pyproject.toml`'un `dev` extra'sındaki `pillow`).
Çıktı, `src/imgflow/cli.py`'nin pencere/görev çubuğu ikonu ve açılış (splash) ekranı için
`Path(__file__)`'a göre okuduğu `src/imgflow/resources/icons/` altına yazılır -- hem
`pip install -e .` hem normal kurulumda güvenilir şekilde bulunur. Splash ekranı AYNI PNG'yi
kullandığından, buradaki renk/ton değişikliği splash'a da otomatik yansır.

Yeniden çalıştırmak (ör. renk/yazı tipi değiştirmek için) güvenlidir, var olan dosyaların
üzerine yazar. Kullanım: `python packaging/generate_icon.py` (venv'de `pillow` kurulu olmalı,
bkz. `pip install -e ".[dev]"`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "imgflow" / "resources" / "icons"
_CANVAS_SIZE = 1024
_CORNER_RADIUS_RATIO = 0.22  # modern "squircle" hissi -- iOS/Windows 11 uygulama ikonlarına yakın
_ICO_SIZES = [16, 32, 48, 64, 128, 256]

# Metalik gri fırçalanmış-çelik hissi için sol-üstten sağ-alta doğru açık gümüşten koyu
# grafite geçen bir gradyan -- tamamen gri tonlarında (renk YOK), "siyah beyaz" isteğine uyar.
_GRADIENT_LIGHT = (205, 206, 209)
_GRADIENT_DARK = (55, 56, 60)
_TEXT_COLOR = (255, 255, 255, 255)
_SHADOW_COLOR = (0, 0, 0, 170)
_SHADOW_BLUR_RADIUS = 14
_SHADOW_OFFSET = (0, 10)
"""Metnin hafif bulanık siyah gölgesi -- gradyanın AÇIK (üst-sol) ucunda bile beyaz metnin
okunabilir kalmasını sağlar (düz beyaz metin, açık gümüş zeminde tek başına kontrastsız
kalırdı)."""

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",  # Segoe UI Bold -- Windows'un varsayılan arayüz fontu
    r"C:\Windows\Fonts\arialbd.ttf",  # Arial Bold -- yaygın yedek
]


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    # Hiçbir TrueType font bulunamazsa (Windows dışı bir ortam) PIL'in dahili bitmap fontuna
    # düşülür -- çirkin ama script en azından ÇÖKMEZ.
    return ImageFont.load_default()


def _metallic_gradient(size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    # Sol-üst köşede 0, sağ-alt köşede 1 olan diyagonal karışım oranı.
    t = (xx.astype(np.float32) + yy.astype(np.float32)) / (2.0 * (size - 1))
    light = np.array(_GRADIENT_LIGHT, dtype=np.float32)
    dark = np.array(_GRADIENT_DARK, dtype=np.float32)
    rgb = light[None, None, :] * (1.0 - t[..., None]) + dark[None, None, :] * t[..., None]
    alpha = np.full((size, size, 1), 255.0, dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=2).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=radius, fill=255)
    return mask


def _draw_logo() -> Image.Image:
    size = _CANVAS_SIZE
    gradient = _metallic_gradient(size)
    mask = _rounded_mask(size, int(size * _CORNER_RADIUS_RATIO))

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(gradient, (0, 0), mask)

    text = "EK"
    font = _load_bold_font(int(size * 0.46))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = (
        (size - text_w) / 2 - bbox[0],
        (size - text_h) / 2 - bbox[1],
    )

    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_pos = (position[0] + _SHADOW_OFFSET[0], position[1] + _SHADOW_OFFSET[1])
    ImageDraw.Draw(shadow_layer).text(shadow_pos, text, font=font, fill=_SHADOW_COLOR)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(_SHADOW_BLUR_RADIUS))
    canvas.alpha_composite(shadow_layer)

    ImageDraw.Draw(canvas).text(position, text, font=font, fill=_TEXT_COLOR)
    return canvas


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logo = _draw_logo()

    png_path = _OUTPUT_DIR / "ek_logo.png"
    logo.save(png_path)

    ico_path = _OUTPUT_DIR / "ek_icon.ico"
    logo.save(ico_path, format="ICO", sizes=[(s, s) for s in _ICO_SIZES])

    print(f"Yazıldı: {png_path}")
    print(f"Yazıldı: {ico_path}")


if __name__ == "__main__":
    main()
