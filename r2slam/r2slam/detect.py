#!/usr/bin/env python3
"""COCO object detection, shared by both data sources.

Model: **YOLO11n**, exported to ONNX at 640x640 by
`vendor/detector/_export/export_yolo11n.py`.  It replaced the ONNX-zoo
`tiny-yolov3-11` on 2026-10-03 for a measured reason, not a fashion one: on the
simulator's own camera frames the old model reported 25 person instances where
`/gazebo/model_states` says there are **three**, and it reported the same wrong
place up to 42 times, so the "seen at least twice" gate could not filter them.
The object it was most persistent about is the `youbot` parked at (4.32, -6.16),
an orange-and-white mobile manipulator.  A stronger backbone fixes the cause.

Two properties still matter and both still hold:

  * **no PyTorch at run time.**  onnxruntime (~15 MB of wheels) or OpenCV.  The
    brief asks for something that fits an embedded platform, and dragging a
    2 GB torch install onto the target is the opposite of that.  torch is used
    once, offline, only to *produce* the .onnx -- it never enters the pipeline,
    the delivery environment or the container.
  * **the decode is checked against the reference implementation.**  Dropping
    tiny-yolov3 means we now own the anchor decode (YOLO11's ONNX output is the
    raw `[1, 84, 8400]`: 4 box + 80 class, no objectness, no NMS).  A wrong
    decoder fails silently and yields plausible-but-mislocated boxes, so
    `tools/check_onnx_parity.py` runs ultralytics' own predictor and this
    decoder over the same frames and asserts the boxes agree to a fraction of a
    pixel.  That script is the reason this file is allowed to do arithmetic on
    raw tensors.

The network is a **COCO-80** detector and is used as-is: nothing here is
trained or fine-tuned.  Which of the 80 classes end up in the map is decided by
the label table in `r2slam/labels.py` (physical gates, not tuned weights), and
`detect_mapped()` returns the project's own label id so the rest of the stack
never has to know about COCO.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # imported as a package (the normal case)
    from . import labels as L
except ImportError:                     # run as a loose script
    import labels as L                  # type: ignore

PERSON_CLASS = 0
COCO80 = None

# The model the pipeline uses when `--model` is not given.  One constant, read
# by `online_pipeline.py`, `replay_explore.py` and this file's own CLI, so
# swapping the detector is one edit rather than three that can drift -- the two
# previous detector swaps each shipped a stale default in one of the three.
# It is a *baked-vocabulary* export (`tools/fetch_yoloworld.py`): its class
# indices are our own prompt order, not COCO's, and the mapping lives in the
# `.classmap.json` sidecar beside it.
DEFAULT_MODEL_FILE = "yoloworld_alias_320.onnx"

# Sentinel for `OnnxYoloDetector(class_map=...)`: "not specified" means "load the
# sidecar beside the model if there is one", which is the only safe default.
# An explicit `None` still means "treat this as a COCO model", so the escape
# hatch survives.
_AUTO_CLASS_MAP = object()

# Letterbox padding colour.  Must match ultralytics' `LetterBox.padding_value`,
# because a different border changes the model's input and therefore its boxes.
PAD_VALUE = 114


def _iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    br = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(ar + br - inter, 1e-9))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS over (N,4) xyxy boxes.  Returns kept indices, best first.

    Written out rather than taken from a library because the two runtimes this
    file supports disagree about whether they have one, and because the graph's
    own suppression layer has already been caught returning an empty index
    tensor while a 0.82 person sat in the frame.  Never trust an in-graph
    post-process that has not been shown to fire.
    """
    xy = np.asarray(boxes, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-sc)
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        a, b = xy[i], xy[rest]
        ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
        ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        br = (np.clip(b[:, 2] - b[:, 0], 0, None)
              * np.clip(b[:, 3] - b[:, 1], 0, None))
        iou = inter / np.maximum(ar + br - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def load_names(path: str) -> List[str]:
    global COCO80
    with open(path, encoding="utf-8") as fh:
        COCO80 = [l.strip() for l in fh if l.strip()]
    return COCO80


def load_sidecar(onnx_path: str) -> Optional[dict]:
    """The whole sidecar dict, or None.  See `load_class_map` for what it is."""
    if not onnx_path:
        return None
    side = os.path.splitext(str(onnx_path))[0] + ".classmap.json"
    if not os.path.isfile(side):
        return None
    try:
        with open(side, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                              # noqa: BLE001
        return None


def load_class_map(onnx_path: str) -> Optional[Dict[int, int]]:
    """`index -> project label` sidecar for a baked-vocabulary model.

    Written by `tools/fetch_yoloworld.py`.  Returns None when there is no
    sidecar, which is the correct answer for a COCO model and also the safe one:
    `detect_mapped` falls back to `labels.COCO_TO_LABEL`, so a missing file
    produces the old behaviour rather than a silent relabelling.
    """
    d = load_sidecar(onnx_path) or {}
    try:
        m = d.get("index_to_label") or {}
        return {int(k): int(v) for k, v in m.items()} or None
    except Exception:                              # noqa: BLE001
        return None


class OnnxYoloDetector:
    """One ONNX COCO detector, two graph dialects.

    `kind` is decided from the graph's own output signature, never from the
    filename, because a mis-named file is exactly the sort of thing that would
    otherwise be discovered as "zero detections, no error":

      * ``"v8"`` -- YOLOv8/v11.  A single raw tensor `[1, 84, 8400]`.  Boxes are
        **cx, cy, w, h in letterboxed input pixels**, classes follow with no
        objectness term.  Decoding, letterbox removal and NMS are ours.
      * ``"v3"`` -- tiny-yolov3 from the ONNX model zoo.  Two tensors, the box
        tuple is **(y1, x1, y2, x2) already in original pixels**, and scores
        arrive either as (80, N) or (N, 80) depending on the runtime.  Kept
        because it is a useful cross-check and because `--model` can still
        point at it; it is no longer the default.

    Both runtimes are still supported, for the reason they always were: neither
    is available everywhere.

      * `onnxruntime`  -- the standard, and now the only one that can run the
                          v11 graph here (it uses SiLU and friends that the
                          local OpenCV 4.8 cannot parse).
      * `cv2.dnn`      -- kept as a fallback and for the v3 graph.

    `backend="auto"` picks whichever works, so the same file runs on both.
    """

    def __init__(self, model: str, conf: float = 0.30, size: Optional[int] = None,
                 backend: str = "auto", num_threads: int = 2,
                 class_map=_AUTO_CLASS_MAP):
        self.conf = float(conf)
        self.model_path = str(model)
        # Prefilter for the raw anchor tensor.  It must be **no higher than the
        # lowest per-class floor in the label table**, otherwise a box that
        # clears its own class floor could be thrown away before any per-class
        # gate ever sees it.  Derived rather than configured so the two cannot
        # drift apart.
        self.prefilter = min(self.conf, float(min(L.LABEL_MIN_CONF.values())))
        self.size = size
        self.backend = None
        self.sess = None
        self.net = None
        self.in_names: List[str] = []
        self.num_threads = int(num_threads)
        self.threads_applied = None
        self.kind = None
        self.last_shapes = None
        # bookkeeping for detect_mapped: how many candidate boxes the network's
        # top-1 class was one we have no name for (see its docstring)
        self.n_unmapped = 0
        # `class index -> project label`, for detectors whose class list is not
        # COCO.  A YOLO-World export produced by `tools/fetch_yoloworld.py` has
        # OUR prompts baked in, in our order, so its index k is the label sidecar
        # in the .classmap.json next to the .onnx -- and applying
        # `COCO_TO_LABEL` to it would relabel every box silently.  None means
        # "this is a COCO model", which is the case for yolo26n.onnx.
        #
        # Loading the sidecar **by default** rather than asking callers to
        # remember: two call sites (`online_pipeline.py`, then `replay_explore.py`)
        # each shipped without the argument, and each therefore silently dropped
        # every class but `person` the moment a baked model became the default.
        # The failure is invisible -- boxes still appear, the log is healthy --
        # so the safe direction is to make the correct behaviour the one you get
        # by doing nothing.  A caller that really wants the COCO mapping passes
        # an explicit `None`.
        if class_map is _AUTO_CLASS_MAP:
            class_map = load_class_map(model)
        self.class_map: Optional[Dict[int, int]] = (
            {int(k): int(v) for k, v in class_map.items()}
            if class_map else None)

        order = ["onnxruntime", "cv2"] if backend == "auto" else [backend]
        errs = []
        for b in order:
            try:
                if b == "onnxruntime":
                    import onnxruntime as ort
                    # Cap the intra-op pool.  Left alone, onnxruntime sizes it
                    # from the visible core count -- 80 here -- while the
                    # container's cgroup quota is 8 CPUs.  The pool then
                    # over-subscribes the quota, and when it runs out the WHOLE
                    # cgroup is frozen until the next period, single-threaded
                    # numpy included.  Measured on this box: the ground-plane
                    # fit (pure numpy, no torch, no ORT) took 19.9 ms/frame with
                    # the detector disabled and 104 ms/frame with it enabled --
                    # a 5x slowdown of code that does not use onnxruntime at
                    # all.  Same lesson as `registration.max_num_threads` in
                    # KISS-ICP: never let a library size its own pool from
                    # `nproc` inside a quota-limited container.
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = max(1, self.num_threads)
                    so.inter_op_num_threads = 1
                    # Busy-waiting threads keep burning CPU after the work is
                    # done; in a quota-limited container that is exactly the
                    # "we have stopped working but we are still charged" state
                    # that ends in a frozen cgroup.  Ask for spinning off.
                    for key in ("session.intra_op.allow_spinning",
                                "session.inter_op.allow_spinning"):
                        try:
                            so.add_session_config_entry(key, "0")
                        except Exception:      # noqa: BLE001 - older runtimes
                            pass
                    self.sess = ort.InferenceSession(
                        model, so, providers=["CPUExecutionProvider"])
                    self.in_names = [i.name for i in self.sess.get_inputs()]
                    self.out_names = [o.name for o in self.sess.get_outputs()]
                    self.threads_applied = so.intra_op_num_threads
                else:
                    import cv2
                    # Same reason as the onnxruntime branch above: OpenCV sizes
                    # its pool from the core count too.
                    cv2.setNumThreads(max(1, self.num_threads))
                    self.net = cv2.dnn.readNetFromONNX(model)
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.threads_applied = max(1, self.num_threads)
                    self.in_names = ["input_1", "image_shape"]
                    self.out_names = list(self.net.getUnconnectedOutLayersNames())
                self.backend = b
                break
            except Exception as exc:            # noqa: BLE001  - try the next one
                errs.append("%s: %s" % (b, exc))
        if self.backend is None:
            raise RuntimeError("no usable ONNX runtime -- " + " | ".join(errs))

        # ---- graph dialect, from the output signature -----------------------
        # (1, 84, 8400) is v8/v11; anything with two outputs is the v3 zoo
        # graph.  Read the *shape* rather than the name so a renamed file does
        # not silently take the wrong decoder.
        if len(self.out_names) >= 2:
            self.kind = "v3"
        else:
            self.kind = "v8"
        if self.size is None:
            # Read the side the graph declares instead of assuming 640.  A
            # baked-vocabulary export is routinely re-made at a smaller side to
            # buy back frame budget, and guessing here feeds a 640 blob to a 320
            # graph -- which onnxruntime rejects loudly, but a laxer runtime
            # would have accepted with every box mis-scaled.  Same principle as
            # `kind` above: the graph's own shape is the authority, never the
            # filename and never a constant.
            self.size = (self._declared_side()
                         or (640 if self.kind == "v8" else 416))
        # v3 needs the (h, w) tensor as a second input; v8 has a single input
        self._shape_name = (self.in_names[1]
                            if self.kind == "v3" and len(self.in_names) > 1 else None)

    # ------------------------------------------------------------------ input
    def _declared_side(self) -> Optional[int]:
        """The square input side the graph declares, or None if it cannot be
        established.

        Two sources, in order of authority:

        1. the graph itself, via onnxruntime's input signature -- the shape the
           session will actually enforce;
        2. failing that, the **sidecar's `imgsz`**, which is the side the export
           was made at (`tools/fetch_yoloworld.py` records it).

        The second exists because `cv2.dnn` cannot be asked for an input shape
        before a forward pass, and the old code answered that with a constant.
        Measured in the simulator container on 2026-10-09: onnxruntime 1.16.3
        refuses the opset-20 export, the detector silently falls back to cv2,
        `_declared_side()` returned None and the pipeline reported `input_px`
        640 for a **320** graph.  cv2 accepted the oversized blob without
        erroring -- the boxes come out plausible and misplaced, which is the
        exact failure the old comment here predicted and did not prevent.

        Returning None still means "guess from the dialect", so a COCO model
        with neither a session shape nor a sidecar behaves exactly as before.
        """
        if self.backend == "onnxruntime":
            try:
                shp = self.sess.get_inputs()[0].shape
            except Exception:                       # noqa: BLE001 - old runtimes
                shp = None
            if (shp and len(shp) == 4
                    and all(isinstance(d, int) and d > 0 for d in shp[1:])
                    and shp[2] == shp[3]):
                return int(shp[2])
        try:
            imgsz = (load_sidecar(self.model_path) or {}).get("imgsz")
            if imgsz and int(imgsz) > 0:
                return int(imgsz)
        except Exception:                           # noqa: BLE001
            pass
        return None

    @staticmethod
    def letterbox_meta(h: int, w: int, size: int) -> Tuple[float, int, int, int, int]:
        """(ratio, new_w, new_h, pad_x, pad_y) for an aspect-preserving pad.

        Same arithmetic as ultralytics' `LetterBox` with `auto=False`: scale by
        `r = min(size/h, size/w)`, centre the result on a `size x size` canvas,
        pad with 114.  Plain `cv2.resize` to a square would stretch 16:9 by a
        factor of 1.78 in y -- a person becomes a tall thin smear, which is a
        plausible reason the old detector hallucinated people in the first
        place.
        """
        r = min(size / float(h), size / float(w))
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (size - nw) // 2, (size - nh) // 2
        return r, nw, nh, px, py

    def preprocess(self, bgr: np.ndarray):
        """BGR frame -> (blob, meta).  `meta` undoes the letterbox."""
        import cv2
        h, w = bgr.shape[:2]
        r, nw, nh, px, py = self.letterbox_meta(h, w, self.size)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), PAD_VALUE, np.uint8)
        canvas[py:py + nh, px:px + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        meta = {"r": r, "pad_x": px, "pad_y": py, "h": h, "w": w}
        return blob, meta

    # ----------------------------------------------------------------- output
    def _run(self, blob, meta):
        """Run the graph, return its raw outputs as a list of ndarrays."""
        if self.backend == "onnxruntime":
            feed = {self.in_names[0]: blob}
            if self._shape_name:
                # v3 only: the zoo graph takes the original (h, w) as (1, 2)
                feed[self._shape_name] = np.array([[meta["h"], meta["w"]]],
                                                  dtype=np.float32)
            return list(self.sess.run(None, feed))
        self.net.setInput(blob, self.in_names[0])
        if self._shape_name:
            # cv2.dnn wants the shape as (1, 1, 2); passing (1, 2) trips
            # "buf.shape() == m.shape()" inside forwardGraph.
            ishape = np.array([[meta["h"], meta["w"]]], dtype=np.float32)
            self.net.setInput(ishape.reshape(1, 1, 2), self._shape_name)
        return list(self.net.forward(self.out_names))

    def raw_candidates(self, bgr: np.ndarray):
        """(boxes, scores) -- every anchor the graph is even slightly sure of.

        `boxes` is (N, 4) **xyxy in ORIGINAL image pixels**, `scores` is
        (80, N) -- always the same convention for both dialects, so nothing
        downstream of this method has to know which graph it is talking to.
        Anchors whose best class is below `self.prefilter` are dropped here,
        which is what keeps the (80, N) array small (8400 x 80 floats would
        otherwise be built every frame).  `prefilter` is derived from the label
        table, never higher than the lowest per-class floor, so no box can be
        lost before its own gate runs.

        The axis convention for `scores` is normalised to (80, N) on purpose:
        onnxruntime hands back (80, N) and cv2.dnn (N, 80) for the v3 graph, and
        reading the wrong axis yields **zero detections instead of an error**.
        """
        h, w = bgr.shape[:2]
        blob, meta = self.preprocess(bgr)
        outs = self._run(blob, meta)
        if self.kind == "v8":
            return self._decode_v8(outs[0], meta)
        return self._decode_v3(outs, h, w)

    def _decode_v8(self, raw: np.ndarray, meta: Dict):
        p = np.asarray(raw)
        if p.ndim == 3:
            p = p[0]
        # (84, 8400) -> (8400, 84).  Decided by which axis is the class axis,
        # not by a hard-coded 84: a different export (e.g. with objectness, or a
        # 91-class set) must not silently decode as garbage.
        if p.shape[0] < p.shape[1]:
            p = p.T
        n_ch = p.shape[1]
        boxes_in = p[:, :4].astype(np.float32)         # cx, cy, w, h (input px)
        if n_ch == 85:                                 # v5/v7-style objectness
            cls = p[:, 5:].T * p[:, 4]
        else:
            cls = p[:, 4:].T
        best = cls.max(axis=0)
        keep = best >= self.prefilter
        if not np.any(keep):
            return np.zeros((0, 4), np.float32), np.zeros((80, 0), np.float32)
        b = boxes_in[keep]
        cls = cls[:, keep]

        r, px, py = meta["r"], meta["pad_x"], meta["pad_y"]
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - bw / 2.0 - px) / r
        y1 = (cy - bh / 2.0 - py) / r
        x2 = (cx + bw / 2.0 - px) / r
        y2 = (cy + bh / 2.0 - py) / r
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        np.clip(xyxy[:, [0, 2]], 0.0, meta["w"], out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0.0, meta["h"], out=xyxy[:, [1, 3]])
        return xyxy.astype(np.float32), cls.astype(np.float32)

    @staticmethod
    def _decode_v3(outs, h: int, w: int):
        boxes = np.asarray(outs[0])[0]
        scores = np.asarray(outs[1])[0]
        if scores.ndim == 2 and scores.shape[0] != 80 and scores.shape[1] == 80:
            scores = scores.T
        if boxes.shape[0] == 0:
            return np.zeros((0, 4), np.float32), np.zeros((80, 0), np.float32)
        n = min(boxes.shape[0], scores.shape[1])
        boxes, scores = boxes[:n], scores[:, :n]
        # (y1, x1, y2, x2), already in original pixels -- multiply nothing.
        xyxy = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
                        axis=1).astype(np.float32)
        np.clip(xyxy[:, [0, 2]], 0.0, w, out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0.0, h, out=xyxy[:, [1, 3]])
        return xyxy, scores.astype(np.float32)

    # ------------------------------------------------------------- public API
    def detect(self, bgr: np.ndarray, want_class: Optional[int] = PERSON_CLASS,
               iou: float = 0.45) -> List[Tuple[int, int, int, int, float, int]]:
        """Returns [(x1, y1, x2, y2, score, class_id)] in ORIGINAL pixels."""
        return self._select(bgr, want_class, iou, self.conf)

    def detect_all(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        return self._select(bgr, None, 0.45, self.conf)

    def _select(self, bgr, want_class, iou, floor,
                classes_of_interest: Optional[Dict[int, int]] = None):
        """Shared tail of every public detect*: score -> gate -> NMS."""
        boxes, scores = self.raw_candidates(bgr)
        self.last_shapes = (boxes.shape, scores.shape)
        n = boxes.shape[0]
        if n == 0:
            return []
        if want_class is None:
            top = np.argmax(scores, axis=0)
            s = scores[top, np.arange(n)]
            cid = top
        else:
            cid = np.full(n, want_class, np.int64)
            s = scores[want_class] if want_class < scores.shape[0] \
                else np.zeros(n, np.float32)
        if classes_of_interest is None:
            m = s >= floor
        else:
            # one floor per class, from the label table
            fl = np.array([classes_of_interest.get(int(c), floor) for c in cid],
                          np.float32)
            m = s >= fl
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            return []
        # Clipping to the image can collapse a box that lay entirely outside it
        # (x1 == x2 after the clamp).  Those carry no information and, worse,
        # survive NMS unscathed -- IoU with everything is zero.
        bx = boxes[idx]
        good = (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])
        idx = idx[good]
        if len(idx) == 0:
            return []
        idx = idx[np.argsort(-s[idx])]
        keep = nms(boxes[idx], s[idx], iou)
        out = []
        for k in keep:
            j = idx[k]
            bx = boxes[j]
            out.append((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]),
                        float(s[j]), int(cid[j])))
        return out

    def detect_mapped(self, bgr: np.ndarray, iou: float = 0.45,
                      min_frac: Optional[dict] = None
                      ) -> List[Tuple[int, int, int, int, float, int]]:
        """One forward pass, every class this project can *name*.

        Returns `[(x1, y1, x2, y2, score, label)]` where `label` is **our** id
        from `r2slam/labels.py`, not a COCO index.

        The selection rule is stated rather than hidden:

        * take the detector's own **top-1** class for the box -- never invent a
          class the network did not prefer;
        * keep it only if the project has a name for that COCO class;
        * apply that class's own confidence floor and minimum box size from the
          label table, and finally suppress overlapping boxes **within the
          surviving set** (two classes can both be right about one object, but
          the map wants one marker).

        The tempting alternative -- take the best *mapped* class regardless of
        what top-1 was -- recovers objects the network confused with an unmapped
        class, but it also turns "bicycle 0.9 / chair 0.2" into a reported chair.
        A semantic map with confident wrong labels is worse than one with fewer
        labels.  `self.n_unmapped` counts what this rule discarded, so the
        trade-off is visible in the run report instead of being a silent choice.
        """
        h = bgr.shape[0]
        boxes, scores = self.raw_candidates(bgr)
        self.last_shapes = (boxes.shape, scores.shape)
        n = boxes.shape[0]
        if n == 0:
            self.n_unmapped = 0
            return []
        top = np.argmax(scores, axis=0)
        top_s = scores[top, np.arange(n)]

        frac = min_frac or {}
        cand = []                       # (box, label, score)
        self.n_unmapped = 0
        for j in range(n):
            coco = int(top[j])
            lab = (self.class_map.get(coco) if self.class_map is not None
                   else L.COCO_TO_LABEL.get(coco))
            if lab is None:
                # Count only boxes the mapped classes would also have had to
                # clear: `self.conf` is the lowest floor in the table, so this
                # is "a confident box we could not name", not "one of the ten
                # thousand near-zero boxes every frame carries".  Without that
                # filter the number is meaningless (it is ~10k/frame).
                if float(top_s[j]) >= self.conf:
                    self.n_unmapped += 1
                continue
            s = float(top_s[j])
            if s < L.LABEL_MIN_CONF.get(lab, self.conf):
                continue
            bx = boxes[j]
            if bx[2] <= bx[0] or bx[3] <= bx[1]:
                continue
            # the box has to be a sane fraction of the frame height -- this is
            # what rejects confident speculation on distant clutter
            if (bx[3] - bx[1]) < frac.get(lab, L.LABEL_MIN_BOX_FRAC[lab]) * h:
                continue
            cand.append((bx, lab, s))
        if not cand:
            return []

        # Suppress across classes as well.  The round-2 code did NMS inside
        # `detect()` only, so `detect_mapped` could hand the same object to the
        # registry twice under two labels (a table reported as both
        # "dining table" and "chair"), and each copy then got its own instance.
        cand.sort(key=lambda c: -c[2])
        arr = np.array([c[0] for c in cand], np.float32)
        sc = np.array([c[2] for c in cand], np.float32)
        keep = nms(arr, sc, iou)
        return [(int(arr[k][0]), int(arr[k][1]), int(arr[k][2]), int(arr[k][3]),
                 float(sc[k]), int(cand[k][1])) for k in keep]


def box_colour(label_id: int):
    """BGR for one of our label ids -- the same colour the map and pcd use."""
    return L.bgr(label_id)


def draw(bgr: np.ndarray, boxes: Sequence, names: Optional[List[str]] = None,
         scale: int = 1, colour=(0, 255, 0), colors=None,
         use_label_table: bool = False) -> np.ndarray:
    """Annotate boxes.

    Two naming schemes are supported on purpose.  `names` indexes a COCO name
    list (that is what the standalone `detect.py` CLI still wants), while
    `use_label_table` interprets `b[5]` as **our** label id and colours the box
    from the shared table -- so a chair is the same colour in the picture, in
    the map panel and in the pcd.  Keeping both avoids a fork of this function.
    """
    import cv2
    out = (cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                      interpolation=cv2.INTER_LINEAR) if scale > 1 else bgr.copy())
    for b in boxes:
        x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
        cid = b[5] if len(b) > 5 else 0
        if use_label_table:
            label = L.name(cid)
            col = L.bgr(cid)
        else:
            label = (names[cid] if names and cid < len(names) else str(cid))
            col = (colors or {}).get(cid, colour)
        cv2.rectangle(out, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale),
                      col, max(1, scale))
        txt = "%s %.2f" % (label, s)
        cv2.putText(out, txt, (x1 * scale, max(11, y1 * scale - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, col, max(1, scale))
    return out


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import time

    import cv2

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "..", "..", "vendor",
                                                    "detector",
                                                    DEFAULT_MODEL_FILE))
    ap.add_argument("--names", default=os.path.join(here, "..", "..", "vendor",
                                                    "detector", "coco.names"))
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--size", type=int, default=0, help="0 = graph default")
    ap.add_argument("--all-classes", action="store_true")
    a = ap.parse_args()

    names = load_names(a.names) if os.path.exists(a.names) else None
    det = OnnxYoloDetector(a.model, conf=a.conf, size=a.size or None)
    print("backend %s  kind %s  size %d  outputs %s"
          % (det.backend, det.kind, det.size, det.out_names))
    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit("no files matched %r" % a.glob)

    tot_ms = 0.0
    for p in files:
        img = cv2.imread(p)
        if img is None:
            print("  SKIP unreadable", p)
            continue
        t0 = time.time()
        boxes = det.detect_all(img) if a.all_classes else det.detect(img)
        ms = (time.time() - t0) * 1e3
        tot_ms += ms
        h, w = img.shape[:2]
        hs = [b[3] - b[1] for b in boxes]
        print("  %-26s %4dx%-4d  %5.1f ms  n=%d  h_px=%s"
              % (os.path.basename(p), w, h, ms, len(boxes), hs))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            d = draw(img, boxes, names, scale=a.scale)
            cv2.imencode(".png", d)[1].tofile(os.path.join(a.out, os.path.basename(p)))
    print("  mean %.1f ms/frame over %d frames" % (tot_ms / len(files), len(files)))
