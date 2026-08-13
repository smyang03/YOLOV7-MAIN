import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


def letterbox(im, size=640):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    out[dh:dh + nh, dw:dw + nw] = resized
    return out, r, dw, dh


def iou_one(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(boxes, scores, classes, iou_thr=0.65):
    keep = []
    for c in np.unique(classes):
        ids = np.where(classes == c)[0]
        order = ids[np.argsort(scores[ids])[::-1]]
        while len(order):
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            order = order[1:][iou_one(boxes[i], boxes[order[1:]]) < iou_thr]
    return np.asarray(keep, dtype=np.int64)


def load_labels(path, w, h):
    p = path.with_suffix('.txt')
    if not p.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        v = line.split()
        if len(v) >= 5:
            c, x, y, bw, bh = map(float, v[:5])
            rows.append([c, (x - bw / 2) * w, (y - bh / 2) * h,
                         (x + bw / 2) * w, (y + bh / 2) * h])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


class Engine:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        # YOLOv7 exports may expose decoded output plus the three feature maps.
        # Use the decoded [1,25200,5+nc] tensor for evaluation.
        self.output = next((n for n in self.outputs
                            if len(tuple(self.engine.get_tensor_shape(n))) == 3
                            and tuple(self.engine.get_tensor_shape(n))[-1] >= 6),
                           self.outputs[0])
        self.in_shape = tuple(self.engine.get_tensor_shape(self.input))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.output))
        self.host_in = cuda.pagelocked_empty(int(np.prod(self.in_shape)), np.float32)
        self.host_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)), np.float32)
        self.dev_in = cuda.mem_alloc(self.host_in.nbytes)
        self.dev_out = cuda.mem_alloc(self.host_out.nbytes)
        self.other_outputs = []
        for name in self.outputs:
            if name == self.output:
                continue
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            buf = cuda.mem_alloc(size * np.dtype(np.float32).itemsize)
            self.other_outputs.append((name, buf))
        self.stream = cuda.Stream()
        self.ctx.set_tensor_address(self.input, int(self.dev_in))
        self.ctx.set_tensor_address(self.output, int(self.dev_out))
        for name, buf in self.other_outputs:
            self.ctx.set_tensor_address(name, int(buf))

    def run(self, x):
        np.copyto(self.host_in, x.reshape(-1))
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        t0 = time.perf_counter()
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()
        return self.host_out.reshape(self.out_shape).copy(), (time.perf_counter() - t0) * 1000


def collect_predictions(engine, images, conf, nms_iou, size):
    preds, times = [], []
    for idx, path in enumerate(images):
        # cv2.imread on Windows may fail for non-ASCII mapped-drive paths.
        raw = np.fromfile(str(path), dtype=np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        lb, r, dw, dh = letterbox(im, size)
        x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        out, ms = engine.run(x[None])
        z = out[0]
        xywh = z[:, :4]
        obj = z[:, 4]
        cls = z[:, 5:]
        ci = np.argmax(cls, axis=1)
        score = obj * cls[np.arange(len(cls)), ci]
        keep = score >= conf
        xywh, score, ci = xywh[keep], score[keep], ci[keep]
        if len(xywh):
            boxes = np.empty_like(xywh)
            boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            k = nms(boxes, score, ci, nms_iou)
            preds.append((boxes[k], score[k], ci[k]))
        else:
            preds.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)))
        times.append(ms)
        if (idx + 1) % 200 == 0:
            print(f'{idx + 1}/{len(images)}')
    return preds, times


def metrics(images, preds, label_dir, ious=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    gt = []
    for p in images:
        raw = np.fromfile(str(p), dtype=np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        gt.append(load_labels(label_dir / p.name, w, h))
    nc = 17
    ap_all, tp_all, fp_all, npos = [], [], [], 0
    for thr in ious:
        aps = []
        for c in range(nc):
            records = []
            total = 0
            for i, (boxes, scores, classes) in enumerate(preds):
                g = gt[i][gt[i][:, 0] == c, 1:]
                total += len(g)
                used = np.zeros(len(g), dtype=bool)
                ids = np.where(classes == c)[0]
                for j in ids[np.argsort(scores[ids])[::-1]]:
                    hit = -1
                    if len(g):
                        ov = iou_one(boxes[j], g)
                        q = int(np.argmax(ov))
                        if ov[q] >= thr and not used[q]:
                            hit, used[q] = q, True
                    records.append((float(scores[j]), 1 if hit >= 0 else 0))
            if total == 0:
                continue
            records.sort(key=lambda x: x[0], reverse=True)
            t = np.asarray([x[1] for x in records], dtype=np.float32)
            f = 1 - t
            if len(t):
                rec = np.cumsum(t) / total
                prec = np.cumsum(t) / np.maximum(np.cumsum(t) + np.cumsum(f), 1e-9)
                mrec = np.r_[0, rec, 1]
                mpre = np.r_[1, prec, 0]
                for k in range(len(mpre) - 1, 0, -1):
                    mpre[k - 1] = max(mpre[k - 1], mpre[k])
                ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
            else:
                ap = 0.0
            aps.append(ap)
            if thr == 0.5:
                tp_all.extend(t.tolist()); fp_all.extend(f.tolist()); npos += total
        ap_all.append(float(np.mean(aps)) if aps else 0.0)
    tp = sum(tp_all)
    fp = sum(fp_all)
    return {'precision': tp / max(tp + fp, 1e-9), 'recall': tp / max(npos, 1e-9),
            'mAP50': ap_all[0], 'mAP50_95': float(np.mean(ap_all)), 'ap_by_iou': ap_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--nms-iou', type=float, default=0.65)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    image_paths = sorted([p for p in Path(args.images).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    engine = Engine(args.engine)
    preds, times = collect_predictions(engine, image_paths, args.conf, args.nms_iou, args.size)
    result = metrics(image_paths, preds, Path(args.labels))
    result.update({'engine': args.engine, 'images': len(image_paths),
                   'latency_ms_mean': float(np.mean(times)),
                   'latency_ms_median': float(np.median(times)),
                   'latency_ms_p95': float(np.percentile(times, 95))})
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
