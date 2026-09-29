#!/usr/bin/env python3
"""把 before/after 两张同参数渲染图并排拼成一张对比图（带标题条）。
用法: python make_compare.py <before.png> <after.png> <out.png> <title_before> <title_after> <main_title>
"""
import sys
from PIL import Image, ImageDraw

before_p, after_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
t_before = sys.argv[4] if len(sys.argv) > 4 else "BEFORE"
t_after = sys.argv[5] if len(sys.argv) > 5 else "AFTER"
t_main = sys.argv[6] if len(sys.argv) > 6 else ""

b = Image.open(before_p).convert("RGB")
a = Image.open(after_p).convert("RGB")
h = min(b.height, a.height)
b = b.resize((int(b.width * h / b.height), h))
a = a.resize((int(a.width * h / a.height), h))

BAR = 56
TOP = 64 if t_main else 0
W = b.width + a.width + 12
canvas = Image.new("RGB", (W, h + BAR + TOP), (255, 255, 255))
canvas.paste(b, (0, TOP))
canvas.paste(a, (b.width + 12, TOP))

d = ImageDraw.Draw(canvas)
if t_main:
    d.rectangle([0, 0, W, TOP], fill=(28, 32, 38))
    d.text((14, 22), t_main, fill=(255, 255, 255))
d.rectangle([0, TOP, b.width, TOP + BAR], fill=(208, 64, 64))
d.text((14, TOP + 20), t_before, fill=(255, 255, 255))
d.rectangle([b.width + 12, TOP, W, TOP + BAR], fill=(38, 132, 72))
d.text((b.width + 26, TOP + 20), t_after, fill=(255, 255, 255))

canvas.save(out_p)
print("[compare] saved", out_p, canvas.size)
