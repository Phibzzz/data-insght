"""Génère assets/logo.png (logo DATAINSIGHT) avec Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        import matplotlib

        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(str(font_path), 78)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    h = 180
    w = int(200 + font.getlength("DATAINSIGHT") + 20)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Pictogramme : carré arrondi dégradé + barres
    for i in range(160):
        t = i / 159
        color = (int(79 + (14 - 79) * t), int(70 + (165 - 70) * t), int(229 + (233 - 229) * t), 255)
        d.line([(10, 10 + i), (170, 10 + i)], fill=color)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([10, 10, 170, 170], radius=36, fill=255)
    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg.paste(img, (0, 0), mask)
    d = ImageDraw.Draw(bg)
    for x, top in [(42, 108), (76, 78), (110, 52), (144, 90)]:
        d.rounded_rectangle([x - 12, top, x + 12, 146], radius=6, fill=(255, 255, 255, 235))
    d.text((200, 45), "DATAINSIGHT", font=font, fill=(79, 70, 229, 255))
    ROOT.joinpath("assets").mkdir(exist_ok=True)
    bg.save(ROOT / "assets" / "logo.png")
    print("assets/logo.png généré")


if __name__ == "__main__":
    main()
