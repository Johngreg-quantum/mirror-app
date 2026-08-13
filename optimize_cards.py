"""
Resize the three level-card images to max 600px wide, save WebP @ q75
alongside, and overwrite the JPG @ q80 as the fallback.

Same settings as optimize_posters.py — these cards are used interchangeably
with posters (see posterFallback() in new-shell/src/lib/adapters), so they
should be sized to match. The DALL-E originals from gen_images.py are
1024x1792 and around 2.5 MB each, which is far more than any card slot needs.
"""
import os

from PIL import Image

TARGET_W = 600
JPG_Q = 80
WEBP_Q = 75
SRC = 'static'
CARDS = ['beginner-card.jpg', 'intermediate-card.jpg', 'advanced-card.jpg']

before_total = 0
after_jpg = 0
after_webp = 0

print(f'Target: {TARGET_W}px wide, JPG q{JPG_Q}, WebP q{WEBP_Q}\n')

for name in CARDS:
    jpg_path = os.path.join(SRC, name)
    if not os.path.exists(jpg_path):
        print(f'  SKIP {name} (not found)')
        continue

    orig_size = os.path.getsize(jpg_path)
    before_total += orig_size

    img = Image.open(jpg_path)
    img.load()
    orig_dims = img.size

    if img.width > TARGET_W:
        ratio = TARGET_W / img.width
        img = img.resize((TARGET_W, int(img.height * ratio)), Image.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    webp_path = jpg_path.replace('.jpg', '.webp')
    img.save(webp_path, 'WEBP', quality=WEBP_Q, method=6)
    after_webp += os.path.getsize(webp_path)

    img.save(jpg_path, 'JPEG', quality=JPG_Q, optimize=True, progressive=True)
    after_jpg += os.path.getsize(jpg_path)

    print(f'  OK   {name:<24} {orig_dims[0]}x{orig_dims[1]} -> {img.size[0]}x{img.size[1]}'
          f'   {orig_size/1024:>8.1f}KB -> JPG {os.path.getsize(jpg_path)/1024:>6.1f}KB'
          f'  WebP {os.path.getsize(webp_path)/1024:>6.1f}KB')

print()
print(f'Before (JPGs):  {before_total/1024/1024:.2f} MB ({before_total:,} bytes)')
print(f'After  (JPGs):  {after_jpg/1024/1024:.2f} MB ({after_jpg:,} bytes)')
print(f'After  (WebPs): {after_webp/1024/1024:.2f} MB ({after_webp:,} bytes)')
print(f'JPG  reduction: {(1-after_jpg/before_total)*100:.1f}%')
print(f'WebP reduction: {(1-after_webp/before_total)*100:.1f}%')
print(f'On-disk total before: {before_total/1024/1024:.2f} MB, '
      f'after (JPG+WebP): {(after_jpg+after_webp)/1024/1024:.2f} MB')
