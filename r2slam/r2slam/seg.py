"""Instance segmentation head, run **alongside** the detector, not instead of it.

Why a separate module rather than a flag on `OnnxYoloDetector`: the two heads
answer different questions and are kept in separate graphs on purpose.

  * the detector (`yoloworld_alias_320.onnx`) is the **open-vocabulary** one --
    51 baked prompts over 39 classes, three of which are in none of COCO's 80;
  * a segmentation export is a **closed COCO-80** model.  It can outline a
    person, and it cannot name a `gym equipment`.

So swapping one for the other trades M4's coverage for P1/P2/P3's masks.  Running
both is the option that does not trade: measured 2026-10-10 on 20 real frames,
back to back in one session, the raw session times were 35.78 ms (detector) and
13.50 / 13.55 ms (`yolo11n-seg` / `yolo26n-seg` at 320) -- the seg head costs
about a third of the detector it accompanies.

⚠️ What the numbers above are **not**: they are the raw `session.run` only.  The
mask decode (coefficients @ prototypes, crop, threshold) is not in them, and
neither is the 3D use of a mask.  `tools/bench_seg_head.py` says the same thing
in the same words; a decoder cost that has not been measured must not be written
down as if it had.

The layout, which is the one thing that silently goes wrong:

    output0    [1, 4 + nc + 32, N]     4 box, nc class scores, 32 mask coeffs
    output1    [1, 32, mh, mw]         the mask prototypes

`4 + nc` is the detection layout.  Decoding a seg export as `4 + nc` reads the
**32 mask coefficients as class scores** -- the same trap that produced a
confident "the class channel is dead" verdict on a YOLOE export once
(`docs/开放词汇边缘化_第一轮实测_20261009.md` sections 32-37).  So `nc` is
*derived from the tensor shape* here and cross-checked against the class count
the caller declares, and a mismatch is an error rather than a guess.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detect import OnnxYoloDetector, nms
from . import labels as L

# COCO-80 is what every plain seg export in this project is trained on.  Kept as
# a constant rather than hard-coded into the arithmetic so a future 91-class or
# 51-class seg model is a parameter, not a rewrite.
N_COCO = 80


class OnnxSegDetector:
    """One ONNX instance-segmentation graph -> boxes, classes and mask crops.

    The contract downstream is deliberately small:

        {"box": (x1, y1, x2, y2),   # ORIGINAL image pixels, float
         "score": float,
         "coco": int,               # COCO-80 index, the model's own class axis
         "label": int,              # our label id, via `labels.COCO_TO_LABEL`
         "mask": bool (h, w),       # the crop, at prototype resolution
         "mask_box": (x0, y0, x1, y1)}   # where the crop sits, ORIGINAL px

    Masks are returned **cropped and at prototype resolution**, not upsampled to
    the frame.  Two reasons: an upsampled 1600x1296 bool per instance is 2 MB per
    instance per frame for a shape that only ever gets resized again for the
    video; and the 3D consumer does not want frame-resolution pixels either --
    it wants "is this projected point inside this instance", which is a lookup
    in the crop's own frame (`mask_cell()`).
    """

    def __init__(self, model: str, conf: float = 0.35, num_threads: int = 2,
                 nc: int = N_COCO):
        self.model_path = str(model)
        self.conf = float(conf)
        self.num_threads = int(num_threads)
        self.nc = int(nc)
        self.size: Optional[int] = None
        self.protos: Optional[np.ndarray] = None      # (32, mh, mw)
        self.mh = self.mw = 0
        # Same bookkeeping as the detector: confident boxes whose top-1 class
        # this project cannot name.  Reported, so "the seg head sees less" can
        # be told apart from "the seg head is fed less" the same way it can on
        # the detector side.
        self.n_unmapped = 0

        import onnxruntime as ort
        so = ort.SessionOptions()
        # Same reason as `detect.OnnxYoloDetector`: never let a library size its
        # own pool from `nproc` inside a quota-limited container.
        so.intra_op_num_threads = max(1, self.num_threads)
        so.inter_op_num_threads = 1
        for key in ("session.intra_op.allow_spinning",
                    "session.inter_op.allow_spinning"):
            try:
                so.add_session_config_entry(key, "0")
            except Exception:                          # noqa: BLE001
                pass
        self.sess = ort.InferenceSession(
            model, so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        if len(shp) == 4 and shp[2] == shp[3] and isinstance(shp[2], int):
            self.size = int(shp[2])
        self.out_shapes = [tuple(o.shape) for o in self.sess.get_outputs()]
        self._check_layout()

    # ------------------------------------------------------------------ layout
    def _check_layout(self) -> None:
        """Fail loudly on a graph that is not `4 + nc + 32` + prototypes.

        Checked once at construction, on the declared shapes, so the failure
        lands at startup with the two numbers that disagree rather than as an
        `IndexError` in the middle of a run -- or worse, as plausible boxes
        decoded from the wrong columns.
        """
        if self.size is None:
            raise ValueError(
                "seg export has no static square input: %s" % self.out_shapes)
        if len(self.out_shapes) < 2:
            raise ValueError(
                "seg export must have a head and a prototype tensor, got %s"
                % (self.out_shapes,))
        head = self.out_shapes[0]
        if len(head) != 3:
            raise ValueError("seg head is not [1, C, N]: %s" % (head,))
        c = int(head[1])
        got = c - 4 - 32
        if got != self.nc:
            raise ValueError(
                "seg head declares C=%d, i.e. nc=%d after 4 box + 32 mask "
                "coefficients, but %d was expected.  A `4 + nc` model has no "
                "prototype tensor and must not be decoded here."
                % (c, got, self.nc))
        proto = self.out_shapes[1]
        if len(proto) != 4 or int(proto[1]) != 32:
            raise ValueError("mask prototypes are not [1, 32, mh, mw]: %s"
                             % (proto,))
        self.mh, self.mw = int(proto[2]), int(proto[3])

    # ------------------------------------------------------------- inference
    def _run(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        r, nw, nh, px, py = OnnxYoloDetector.letterbox_meta(h, w, self.size)
        import cv2
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = resized
        blob = np.transpose(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            (2, 0, 1))[None]
        outs = self.sess.run(None, {self.in_name: blob})
        return outs, {"r": r, "pad_x": px, "pad_y": py, "h": h, "w": w}

    def detect(self, bgr: np.ndarray, iou: float = 0.45) -> List[Dict]:
        outs, meta = self._run(bgr)
        head = np.asarray(outs[0])
        if head.ndim == 3:
            head = head[0]
        if head.shape[0] < head.shape[1]:
            head = head.T                            # -> (N, C)
        n_ch = head.shape[1]
        if n_ch != 4 + self.nc + 32:
            raise ValueError("head width %d != 4 + %d + 32" % (n_ch, self.nc))

        boxes_in = head[:, :4].astype(np.float32)     # cx, cy, w, h (input px)
        cls = head[:, 4:4 + self.nc].T                # (nc, N)
        best = cls.max(axis=0)
        keep = best >= self.conf
        if not np.any(keep):
            self.n_unmapped = 0
            return []
        kc = cls[:, keep].argmax(axis=0)

        # cxcywh -> xyxy, letterbox undone, in ORIGINAL pixels
        r, pad_x, pad_y = meta["r"], meta["pad_x"], meta["pad_y"]
        b = boxes_in[keep]
        x1 = (b[:, 0] - b[:, 2] / 2 - pad_x) / r
        y1 = (b[:, 1] - b[:, 3] / 2 - pad_y) / r
        x2 = (b[:, 0] + b[:, 2] / 2 - pad_x) / r
        y2 = (b[:, 1] + b[:, 3] / 2 - pad_y) / r
        boxes = np.stack([x1, y1, x2, y2], 1).astype(np.float32)
        scores = best[keep].astype(np.float32)

        # **The same per-class gates the detector uses**, not this class's own
        # looser `conf`.  Without them the two heads would be showing different
        # rules and the picture would answer "which model is more permissive"
        # instead of "what does the mask add".  Unnameable top-1 classes are
        # dropped and counted, exactly as `detect_mapped` does.
        h0 = meta["h"]
        self.n_unmapped = 0
        sur: List[int] = []
        for j in range(len(scores)):
            coco = int(kc[j])
            lab = L.COCO_TO_LABEL.get(coco)
            if lab is None:
                if float(scores[j]) >= self.conf:
                    self.n_unmapped += 1
                continue
            if float(scores[j]) < L.LABEL_MIN_CONF.get(lab, self.conf):
                continue
            if boxes[j][2] <= boxes[j][0] or boxes[j][3] <= boxes[j][1]:
                continue
            if ((boxes[j][3] - boxes[j][1])
                    < L.LABEL_MIN_BOX_FRAC[lab] * h0):
                continue
            sur.append(j)
        if not sur:
            return []

        # Masks are decoded for the NMS survivors only.  Decoding first and
        # discarding later would multiply the prototype matmul by the
        # prefilter's box count, which on a real frame is ~10x the output.
        sub_b = boxes[sur]
        sub_s = scores[sur]
        pick = nms(sub_b, sub_s, iou)
        if not pick:
            return []
        sel = [sur[p] for p in pick]
        coef = head[keep][sel][:, 4 + self.nc:].astype(np.float32)     # (K, 32)
        proto = np.asarray(outs[1])[0].reshape(32, -1)                 # (32, mh*mw)
        m = 1.0 / (1.0 + np.exp(-(coef @ proto)))                      # (K, mh*mw)
        m = m.reshape(len(sel), self.mh, self.mw) > 0.5

        g = self.mw / float(self.size)               # input px -> prototype cell
        out: List[Dict] = []
        for k, i in enumerate(sel):
            coco = int(kc[i])
            bx = boxes_in[keep][i]
            # crop in prototype coords, from the **input-frame** box
            ix0 = int(np.clip(round((bx[0] - bx[2] / 2) * g), 0, self.mw))
            iy0 = int(np.clip(round((bx[1] - bx[3] / 2) * g), 0, self.mh))
            ix1 = int(np.clip(round((bx[0] + bx[2] / 2) * g), 0, self.mw))
            iy1 = int(np.clip(round((bx[1] + bx[3] / 2) * g), 0, self.mh))
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            out.append({
                "box": tuple(float(v) for v in boxes[i]),
                "score": float(scores[i]),
                "coco": coco,
                "label": int(L.COCO_TO_LABEL.get(coco, L.BACKGROUND)),
                "mask": m[k, iy0:iy1, ix0:ix1],
                "mask_box": ((ix0 / g - pad_x) / r, (iy0 / g - pad_y) / r,
                             (ix1 / g - pad_x) / r, (iy1 / g - pad_y) / r),
            })
        return out

    def mask_cell(self, inst: Dict, u: float, v: float, w: int, h: int) -> bool:
        """Is image point `(u, v)` inside `inst`?  `(u, v)` in ORIGINAL pixels.

        This is the only mask query the 3D side needs: project a LiDAR point
        into the image, ask whether it is inside the instance.  Going through
        `mask_box` keeps it a lookup in the crop's own frame, so nothing has to
        be upsampled.
        """
        mx0, my0 = inst["mask_box"][0], inst["mask_box"][1]
        mx1, my1 = inst["mask_box"][2], inst["mask_box"][3]
        if not (mx0 <= u <= mx1 and my0 <= v <= my1):
            return False
        iy = int((v - my0) / max(my1 - my0, 1e-6) * inst["mask"].shape[0])
        ix = int((u - mx0) / max(mx1 - mx0, 1e-6) * inst["mask"].shape[1])
        iy = min(max(iy, 0), inst["mask"].shape[0] - 1)
        ix = min(min(max(ix, 0), inst["mask"].shape[1] - 1), inst["mask"].shape[1] - 1)
        return bool(inst["mask"][iy, ix])


def matches_for_label(insts: Sequence[Dict], label: int) -> List[Dict]:
    """Instances of one project label, best score first."""
    sel = [i for i in insts if int(i["label"]) == int(label)]
    sel.sort(key=lambda i: -float(i["score"]))
    return sel
