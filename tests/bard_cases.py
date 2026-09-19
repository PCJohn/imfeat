"""Canonical and adversarial inputs for the bard bank. Kept out of the main suite."""

import numpy as np


def _rgb(a):  # replicate a 2-D pattern into 3 channels with per-channel offsets
    o = np.stack([a, np.roll(a, 3, 1), np.roll(a, 5, 0)], -1)
    return np.clip(o, 0, 255).astype(np.uint8)


def build():
    C = []
    H = W = 256
    z = np.zeros((H, W), np.uint8)
    C.append(("all_black", _rgb(z), 5, 2))
    C.append(("all_white", _rgb(z + 255), 5, 2))
    C.append(("mid_grey", _rgb(z + 128), 5, 2))
    rng = np.random.RandomState(0)
    C.append(("uniform_noise", rng.randint(0, 256, (H, W, 3)).astype(np.uint8), 5, 2))
    C.append(("low_noise", (rng.randint(124, 132, (H, W, 3))).astype(np.uint8), 5, 2))
    # a horizontal linear ramp: every lag sees one tap up and one down, so nothing fires
    ramp = np.tile(np.linspace(0, 255, W, dtype=np.float64), (H, 1)).astype(np.uint8)
    C.append(("ramp_x", _rgb(ramp), 5, 2))
    # a single step edge: one side matches the centre, so the bar test must score zero
    step = z.copy()
    step[:, W // 2 :] = 255
    C.append(("step_edge", _rgb(step), 5, 2))
    # isolated bars of each lag's width, light-on-dark and dark-on-light
    for wd in (1, 2, 4, 8):
        b = z.copy()
        for x0 in range(10, W - 10, 32):
            b[:, x0 : x0 + wd] = 255
        C.append((f"vbars_w{wd}", _rgb(b), 5, 2))
        C.append((f"vbars_w{wd}_inv", _rgb(255 - b), 5, 2))
        hb = z.copy()
        for y0 in range(10, H - 10, 32):
            hb[y0 : y0 + wd, :] = 255
        C.append((f"hbars_w{wd}", _rgb(hb), 5, 2))
    ck = (np.indices((H, W)).sum(0) % 2 * 255).astype(np.uint8)
    C.append(("checker1", _rgb(ck), 5, 2))
    # content only at the very border, to exercise the scalar border strips
    bd = z.copy()
    bd[:, :3] = 255
    bd[:, -3:] = 255
    bd[:3, :] = 255
    bd[-3:, :] = 255
    C.append(("border_only", _rgb(bd), 5, 2))
    # single lit pixel / single lit row
    sp = z.copy()
    sp[H // 2, W // 2] = 255
    C.append(("one_pixel", _rgb(sp), 5, 2))
    # geometry variations on a stroke-rich image
    rich = z.copy()
    for k in range(60):
        y = rng.randint(4, H - 8)
        x = rng.randint(4, W - 40)
        wd = rng.randint(1, 5)
        rich[y : y + wd, x : x + 36] = 250 if k % 2 else 5
    rich = np.maximum(rich, (rng.rand(H, W) * 40).astype(np.uint8))
    C.append(("strokes_g5_s2", _rgb(rich), 5, 2))
    C.append(("strokes_g5_s1", _rgb(rich), 5, 1))
    C.append(("strokes_g6_s2", _rgb(rich), 6, 2))
    C.append(("strokes_g4_s4", _rgb(rich), 4, 4))
    C.append(("strokes_gray", rich, 5, 2))
    big = _rgb(np.kron(rich, np.ones((2, 2), np.uint8))[:512, :512])
    C.append(("strokes_512", big, 6, 2))
    # cw = 16: the geometry where a cell still holds a whole number of psadbw groups
    # after phase-splitting by the column stride
    C.append(("strokes_512_g5", big, 5, 2))
    C.append(("strokes_512_g5_s4", big, 5, 4))
    C.append(("strokes_512_g4_s2", big, 4, 2))
    # Channel counts other than 3, at a geometry where the phase-split engages
    # (cw / sx >= 8). Every layout has its own gather path; a missing one silently
    # falls back to a scalar byte loop over the whole frame, which is correct but
    # catastrophically slow -- and invisible to a C=3-only test set.
    C.append(("grey_512_g5", big[:, :, 0], 5, 2))
    C.append(("c2_512_g5", np.ascontiguousarray(big[:, :, :2]), 5, 2))
    C.append(("c4_512_g5", np.ascontiguousarray(np.concatenate([big, big[:, :, :1]], -1)), 5, 2))
    C.append(("c2_512_g5_s4", np.ascontiguousarray(big[:, :, :2]), 5, 4))
    # non-square and not-divisible-by-grid geometries
    C.append(("rect_384x256", _rgb(rich[:, :192])[:, :192], 5, 2))
    C.append(("odd_250x250", _rgb(rich[:250, :250]), 5, 2))
    C.append(("thin_24x256", _rgb(rich[:24]), 3, 2))
    C.append(("tiny_8x8", _rgb(rich[:8, :8]), 1, 1))
    return C
