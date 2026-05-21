"""Verify CSS reflection (cloned rows), rocky bg + noise, dark cards, shadow, headline."""
import os, socket, subprocess, sys, time, urllib.request
SERVER="http://127.0.0.1:8765"
def open_port(p):
    s=socket.socket()
    try: s.bind(("127.0.0.1",p)); s.close(); return False
    except OSError: return True
def wait(u,t=20):
    t0=time.time()
    while time.time()-t0<t:
        try:
            with urllib.request.urlopen(u,timeout=1) as r:
                if r.status==200: return True
        except: time.sleep(0.3)
    return False

if open_port(8765): sys.exit("port in use")
proc=subprocess.Popen([sys.executable,"-m","uvicorn","main:app","--host","127.0.0.1","--port","8765","--log-level","warning"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    if not wait(SERVER+"/"): sys.exit("server failed")
    os.makedirs("_smoke_out", exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b=pw.chromium.launch(headless=True)
        for label, vp in [("desktop",{"width":1440,"height":900}), ("mobile",{"width":390,"height":844})]:
            ctx=b.new_context(viewport=vp)
            pg=ctx.new_page()
            errs=[]
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.add_init_script("try{localStorage.removeItem('mirror_token')}catch(e){}")
            pg.goto(SERVER+"/",wait_until="networkidle")
            pg.wait_for_timeout(4000)  # wait past the 800ms clone setTimeout
            pg.screenshot(path=f"_smoke_out/csr_{label}.png", clip={"x":0,"y":0,"width":vp['width'],"height":vp['height']})
            data = pg.evaluate("""() => {
                const reflect = document.getElementById('ml-reflect');
                const reflectInner = document.getElementById('ml-reflect-inner');
                const cloneRows = reflectInner ? reflectInner.querySelectorAll('.ml-mrow').length : 0;
                const cloneCards = reflectInner ? reflectInner.querySelectorAll('.ml-card').length : 0;
                // Confirm no duplicate id="ml-rows-container"
                const dupIds = document.querySelectorAll('[id="ml-rows-container"]').length;
                // Confirm old canvas reflection is gone
                const oldRefl = document.getElementById('ml-reflection-wrap');
                const oldCanvas = document.getElementById('ml-reflection-canvas');
                const landingCS = getComputedStyle(document.getElementById('mirrorLanding'));
                const beforeCS = getComputedStyle(document.getElementById('mirrorLanding'), '::before');
                const card = document.querySelector('.ml-card');
                const cardCS = card && getComputedStyle(card);
                const frame = document.getElementById('ml-frame');
                const frameCS = frame && getComputedStyle(frame);
                const h1 = document.querySelector('#mlCenter h1');
                const h1CS = h1 && getComputedStyle(h1);
                return {
                    reflectPresent: !!reflect,
                    reflectInnerHasClone: cloneRows > 0,
                    cloneRowCount: cloneRows,
                    cloneCardCount: cloneCards,
                    dupRowsContainerIds: dupIds,
                    oldCanvasRemoved: !oldRefl && !oldCanvas,
                    landingBg: landingCS.backgroundImage.slice(0, 260),
                    beforeNoisePresent: beforeCS && beforeCS.backgroundImage.includes('data:image/svg'),
                    beforeOpacity: beforeCS && beforeCS.opacity,
                    cardFilter: cardCS && cardCS.filter,
                    frameShadow: frameCS && frameCS.boxShadow.slice(0, 280),
                    h1Family: h1CS && h1CS.fontFamily,
                    h1Size: h1CS && h1CS.fontSize,
                    h1Spacing: h1CS && h1CS.letterSpacing,
                    htmlOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
                };
            }""")
            print(f"=== {label} ===")
            for k, v in data.items(): print(f"  {k}: {v}")
            print(f"  errors: {errs[:3]}")
            ctx.close()
        b.close()
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except: proc.kill()
