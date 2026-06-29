"""
Resize all /static/posters/*.jpg to max 600px wide,
save WebP @ q75 alongside, overwrite the JPG @ q80 as the fallback.
"""
from PIL import Image, UnidentifiedImageError
import os, glob

TARGET_W = 600
JPG_Q = 80
WEBP_Q = 75
SRC = 'static/posters'

processed = 0
skipped = 0
before_total = 0
after_jpg = 0
after_webp = 0
errors = []

posters = sorted(glob.glob(os.path.join(SRC, '*.jpg')))
print(f'Found {len(posters)} *.jpg files in {SRC}')
print(f'Target: {TARGET_W}px wide, JPG q{JPG_Q}, WebP q{WEBP_Q}\n')

for jpg_path in posters:
    name = os.path.basename(jpg_path)
    orig_size = os.path.getsize(jpg_path)
    before_total += orig_size
    try:
        img = Image.open(jpg_path)
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        print(f'  SKIP {name:<30} ({e.__class__.__name__}: not a valid image, {orig_size} bytes)')
        skipped += 1
        continue

    # Resize maintaining aspect ratio if wider than target
    if img.width > TARGET_W:
        ratio = TARGET_W / img.width
        new_h = int(img.height * ratio)
        img = img.resize((TARGET_W, new_h), Image.LANCZOS)

    # Ensure RGB (some posters may have alpha or palette)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save WebP
    webp_path = jpg_path.replace('.jpg', '.webp')
    img.save(webp_path, 'WEBP', quality=WEBP_Q, method=6)
    after_webp += os.path.getsize(webp_path)

    # Overwrite JPG fallback (smaller)
    img.save(jpg_path, 'JPEG', quality=JPG_Q, optimize=True, progressive=True)
    after_jpg += os.path.getsize(jpg_path)

    new_size = os.path.getsize(jpg_path)
    print(f'  OK   {name:<30}  {orig_size/1024:>7.1f}KB -> JPG {new_size/1024:>5.1f}KB  WebP {os.path.getsize(webp_path)/1024:>5.1f}KB')
    processed += 1

print()
print(f'Processed: {processed}, Skipped: {skipped}')
print(f'Before total (JPGs):       {before_total/1024/1024:.2f} MB ({before_total:,} bytes)')
print(f'After  total (JPGs):       {after_jpg/1024/1024:.2f} MB ({after_jpg:,} bytes)')
print(f'After  total (WebPs):      {after_webp/1024/1024:.2f} MB ({after_webp:,} bytes)')
print(f'JPG  reduction: {(1-after_jpg/before_total)*100:.1f}%')
print(f'WebP reduction: {(1-after_webp/before_total)*100:.1f}%')
