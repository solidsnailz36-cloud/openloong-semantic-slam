#!/usr/bin/env python3
"""COCO person detection, shared by both data sources.

Model: `tiny-yolov3-11.onnx` from the ONNX model zoo.  Two properties make it
the right choice here rather than an ultralytics stack:

  * **no PyTorch.** It runs on onnxruntime (~7 MB of wheels) or OpenCV.  The
    brief asks for something that fits an embedded platform, and dragging a
    2 GB torch install onto the target is the opposite of that.
  * **NMS is inside the graph.** The outputs are already `boxes / scores /
    indices`, so there is no anchor/stride decoding to get subtly wrong --
    which matters because a wrong decoder fails silently and produces
    plausible-but-mislocated boxes.

Output is a list of `(x1, y1, x2, y2, score)` in ORIGINAL image pixels for COCO
class 0 (`person`), ready for the LiDAR/RGB fusion step.

Runtime note: on the cloud box this needs a newer OpenCV/numpy than the ROS
system ones, installed side by side:

    pip3 install --target /root/sim/pylibs --no-deps opencv-python-headless==4.8.1.78
    pip3 install --target /root/sim/pylibs onnxruntime==1.17.3
    PYTHONPATH=/root/sim/pylibs python3 ...
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

PERSON_CLASS = 0
COCO80 = None


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


def load_names(path: str) -> List[str]:
    global COCO80
    with open(path, encoding="utf-8") as fh:
        COCO80 = [l.strip() for l in fh if l.strip()]
    return COCO80


class OnnxYoloDetector:
    """tiny-yolov3 with a fused (but unusable) NMS layer.

    Two runtimes are supported because neither one is available everywhere:

      * `onnxruntime`  -- the standard, but it has no wheel for Python 3.13 yet
                          (the local dev machine) and OpenCV 4.8 on the cloud box
                          cannot even parse this graph (it contains a BOOL
                          constant).
      * `cv2.dnn`      -- needs OpenCV >= 5, which is what the local machine has
                          and what OpenCV 4.8 on the cloud box lacks.

    `backend="auto"` picks whichever works, so the same file runs on both.
    """

    def __init__(self, model: str, conf: float = 0.30, size: int = 416,
                 backend: str = "auto", num_threads: int = 2):
        self.conf = conf
        self.size = size
        self.backend = None
        self.sess = None
        self.net = None
        self.in_names: List[str] = []
        self.num_threads = int(num_threads)
        self.threads_applied = None

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
                    got_in = self.net.getLayerNames()
                    self.in_names = ["input_1", "image_shape"]
                    self.out_names = list(self.net.getUnconnectedOutLayersNames())
                    del got_in
                self.backend = b
                break
            except Exception as exc:            # noqa: BLE001  - try the next one
                errs.append("%s: %s" % (b, exc))
        if self.backend is None:
            raise RuntimeError("no usable ONNX runtime -- " + " | ".join(errs))

        self._shape_name = self.in_names[1] if len(self.in_names) > 1 else None

    def preprocess(self, bgr: np.ndarray):
        import cv2
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        blob = cv2.resize(rgb, (self.size, self.size)).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        return blob, np.array([[h, w]], dtype=np.float32)

    def _run(self, blob, ishape):
        if self.backend == "onnxruntime":
            feed = {self.in_names[0]: blob}
            if self._shape_name:
                feed[self._shape_name] = ishape                 # (1, 2)
            return self.sess.run(None, feed)[:2]
        self.net.setInput(blob, self.in_names[0])
        if self._shape_name:
            # cv2.dnn wants the shape as (1, 1, 2); passing (1, 2) trips
            # "buf.shape() == m.shape()" inside forwardGraph.
            self.net.setInput(ishape.reshape(1, 1, 2), self._shape_name)
        outs = self.net.forward(self.out_names)
        return outs[0], outs[1]

    def detect(self, bgr: np.ndarray, want_class: Optional[int] = PERSON_CLASS,
               iou: float = 0.45) -> List[Tuple[int, int, int, int, float, int]]:
        """Returns [(x1, y1, x2, y2, score, class_id)] in ORIGINAL pixels.

        Two things had to be measured rather than assumed:

        * the box tuple is **(y1, x1, y2, x2)** and is **already in original
          image pixels**.  The first version multiplied by (w, h) as if the
          values were normalised, which turned a 444 px-tall person into a
          -687,832 px-tall one.  Multiply nothing; just reorder.
        * the graph's fused NMS (`yolonms_layer_1:2`) returns an **empty index
          tensor** even when a 0.82 person is present, so the suppression is
          done here instead.  Never trust an in-graph post-process that has not
          been shown to fire.
        """
        h, w = bgr.shape[:2]
        blob, ishape = self.preprocess(bgr)
        boxes, scores = self._run(blob, ishape)
        boxes = boxes[0]
        scores = scores[0]
        # The two runtimes disagree on the score tensor layout: onnxruntime
        # hands back (80, N), cv2.dnn hands back (N, 80).  Normalise to (80, N)
        # -- reading the wrong axis silently yields zero detections instead of
        # an error, which is exactly the kind of bug that costs an afternoon.
        if scores.ndim == 2 and scores.shape[0] != 80 and scores.shape[1] == 80:
            scores = scores.T
        self.last_shapes = (boxes.shape, scores.shape)

        if want_class is None:
            cand = [(j, int(np.argmax(scores[:, j])), float(scores[:, j].max()))
                    for j in range(boxes.shape[0])]
        else:
            cand = [(j, want_class, float(scores[want_class, j]))
                    for j in range(boxes.shape[0])]
        cand = [c for c in cand if c[2] >= self.conf]
        if not cand:
            return []
        cand.sort(key=lambda c: -c[2])

        keep: List[Tuple[int, int, int, int, float, int]] = []
        for j, cid, s in cand:
            y1, x1, y2, x2 = boxes[j]
            cand_box = (max(0.0, x1), max(0.0, y1),
                        min(float(w), x2), min(float(h), y2))
            if cand_box[2] <= cand_box[0] or cand_box[3] <= cand_box[1]:
                continue
            if any(_iou(cand_box, (k[0], k[1], k[2], k[3])) > iou for k in keep):
                continue
            keep.append((int(cand_box[0]), int(cand_box[1]),
                         int(cand_box[2]), int(cand_box[3]), s, cid))
        return keep

    def detect_all(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        return self.detect(bgr, want_class=None)


def draw(bgr: np.ndarray, boxes: Sequence, names: Optional[List[str]] = None,
         scale: int = 1, colour=(0, 255, 0)) -> np.ndarray:
    import cv2
    out = (cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                      interpolation=cv2.INTER_LINEAR) if scale > 1 else bgr.copy())
    for b in boxes:
        x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
        cid = b[5] if len(b) > 5 else 0
        label = (names[cid] if names and cid < len(names) else str(cid))
        cv2.rectangle(out, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale),
                      colour, max(1, scale))
        txt = "%s %.2f" % (label, s)
        cv2.putText(out, txt, (x1 * scale, max(11, y1 * scale - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, colour, max(1, scale))
    return out


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import time

    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny-yolov3-11.onnx")
    ap.add_argument("--names", default="coco.names")
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--all-classes", action="store_true")
    a = ap.parse_args()

    names = load_names(a.names) if os.path.exists(a.names) else None
    det = OnnxYoloDetector(a.model, conf=a.conf)
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
