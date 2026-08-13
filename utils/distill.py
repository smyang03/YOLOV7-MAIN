import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.torch_utils import is_parallel


def load_distillation_teacher(weights, device):
    teacher = attempt_load(weights, map_location=device).to(device).float().eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def detect_module(model):
    model = model.module if is_parallel(model) else model
    return model.model[-1]


def as_raw_outputs(output, nl):
    if isinstance(output, (list, tuple)):
        if len(output) >= 2 and isinstance(output[1], (list, tuple)):
            return list(output[1])[:nl]
        if len(output) >= nl and all(torch.is_tensor(x) for x in output[:nl]):
            return list(output[:nl])
    raise TypeError(f"cannot extract raw detection outputs from {type(output)}")


def parse_cross_stride_pairs(pairs):
    parsed = []
    for pair in pairs or []:
        if isinstance(pair, str):
            text = pair.replace("->", ":").replace(",", ":")
            parts = [p for p in text.split(":") if p]
            if len(parts) != 2:
                raise ValueError(f"invalid cross stride pair '{pair}', expected teacher:student")
            parsed.append((int(parts[0]), int(parts[1])))
        else:
            parsed.append((int(pair[0]), int(pair[1])))
    return parsed


def pool_to_hw(x, hw):
    target_h, target_w = int(hw[0]), int(hw[1])
    if x.shape[2] == target_h and x.shape[3] == target_w:
        return x
    b, na, h, w, c = x.shape
    pooled = x.permute(0, 1, 4, 2, 3).reshape(b * na * c, 1, h, w)
    pooled = F.adaptive_max_pool2d(pooled, (target_h, target_w))
    return pooled.reshape(b, na, c, target_h, target_w).permute(0, 1, 3, 4, 2).contiguous()


class ResponseDistillationLoss(nn.Module):
    def __init__(
        self,
        student_model,
        teacher_model,
        strides=(),
        temperature=1.0,
        conf_thres=0.01,
        obj_weight=1.0,
        cls_weight=1.0,
        box_weight=0.0,
        small_gain=1.0,
        small_px=128.0,
        cross_strides=(),
        cross_weight=1.0,
    ):
        super().__init__()
        self.student_det = detect_module(student_model)
        self.teacher_det = detect_module(teacher_model)
        self.student_strides = [int(s) for s in self.student_det.stride]
        self.teacher_strides = [int(s) for s in self.teacher_det.stride]
        self.teacher_index = {stride: i for i, stride in enumerate(self.teacher_strides)}
        self.strides = set(int(s) for s in strides) if strides else set(self.student_strides)
        self.temperature = max(float(temperature), 1e-6)
        self.conf_thres = float(conf_thres)
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.small_gain = max(float(small_gain), 1.0)
        self.small_px = float(small_px)
        self.cross_stride_pairs = parse_cross_stride_pairs(cross_strides)
        self.cross_weight = float(cross_weight)

    def batch_has_small_target(self, targets, imgs):
        if self.small_gain <= 1.0 or targets is None or targets.numel() == 0:
            return False
        img_px = float(max(imgs.shape[-2:]))
        wh_px = targets[:, 4:6] * img_px
        return bool((wh_px.max(1)[0] <= self.small_px).any().item())

    @staticmethod
    def masked_mean(loss, mask):
        if mask is None:
            return loss.mean()
        mask = mask.to(loss.dtype)
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, student_output, teacher_output, targets=None, imgs=None):
        student_raw = as_raw_outputs(student_output, self.student_det.nl)
        teacher_raw = as_raw_outputs(teacher_output, self.teacher_det.nl)
        student_index = {stride: i for i, stride in enumerate(self.student_strides)}
        temp = self.temperature
        device = student_raw[0].device
        total = torch.zeros((), device=device)
        components = torch.zeros(4, device=device)
        matched = 0

        for student_i, stride in enumerate(self.student_strides):
            if stride not in self.strides or stride not in self.teacher_index:
                continue
            teacher_i = self.teacher_index[stride]
            student = student_raw[student_i]
            teacher = teacher_raw[teacher_i].detach().to(student.device, dtype=student.dtype)
            if student.shape != teacher.shape:
                raise ValueError(
                    f"distill stride {stride} shape mismatch: "
                    f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
                )

            teacher_obj_conf = teacher[..., 4].sigmoid().detach()
            positive_mask = teacher_obj_conf >= self.conf_thres

            obj_loss = F.binary_cross_entropy_with_logits(
                student[..., 4] / temp,
                (teacher[..., 4] / temp).sigmoid(),
                reduction="mean",
            ) * (temp ** 2)

            cls_loss = torch.zeros((), device=device)
            if self.cls_weight > 0.0 and student.shape[-1] > 5:
                cls_raw = F.binary_cross_entropy_with_logits(
                    student[..., 5:] / temp,
                    (teacher[..., 5:] / temp).sigmoid(),
                    reduction="none",
                ) * (temp ** 2)
                cls_loss = self.masked_mean(cls_raw.mean(-1), positive_mask)

            box_loss = torch.zeros((), device=device)
            if self.box_weight > 0.0:
                box_raw = F.smooth_l1_loss(
                    student[..., :4].sigmoid(),
                    teacher[..., :4].sigmoid(),
                    reduction="none",
                ).mean(-1)
                box_loss = self.masked_mean(box_raw, positive_mask)

            layer_loss = (
                self.obj_weight * obj_loss
                + self.cls_weight * cls_loss
                + self.box_weight * box_loss
            )
            total = total + layer_loss
            components += torch.stack((layer_loss.detach(), obj_loss.detach(), cls_loss.detach(), box_loss.detach()))
            matched += 1

        for teacher_stride, student_stride in self.cross_stride_pairs:
            if self.cross_weight <= 0.0:
                continue
            if teacher_stride not in self.teacher_index or student_stride not in student_index:
                continue
            teacher = teacher_raw[self.teacher_index[teacher_stride]].detach()
            student = student_raw[student_index[student_stride]]
            teacher = teacher.to(student.device, dtype=student.dtype)
            if student.shape[:2] != teacher.shape[:2] or student.shape[-1] != teacher.shape[-1]:
                raise ValueError(
                    f"cross distill {teacher_stride}->{student_stride} shape mismatch: "
                    f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
                )

            teacher_obj_prob = pool_to_hw((teacher[..., 4:5] / temp).sigmoid(), student.shape[2:4]).squeeze(-1)
            teacher_cls_prob = pool_to_hw((teacher[..., 5:] / temp).sigmoid(), student.shape[2:4])
            positive_mask = teacher_obj_prob >= self.conf_thres

            obj_loss = F.binary_cross_entropy_with_logits(
                student[..., 4] / temp,
                teacher_obj_prob.detach(),
                reduction="mean",
            ) * (temp ** 2)

            cls_loss = torch.zeros((), device=device)
            if self.cls_weight > 0.0 and student.shape[-1] > 5:
                cls_raw = F.binary_cross_entropy_with_logits(
                    student[..., 5:] / temp,
                    teacher_cls_prob.detach(),
                    reduction="none",
                ) * (temp ** 2)
                cls_loss = self.masked_mean(cls_raw.mean(-1), positive_mask)

            box_loss = torch.zeros((), device=device)
            layer_loss = self.cross_weight * (self.obj_weight * obj_loss + self.cls_weight * cls_loss)
            total = total + layer_loss
            components += torch.stack((layer_loss.detach(), obj_loss.detach(), cls_loss.detach(), box_loss.detach()))
            matched += 1

        if matched == 0:
            raise ValueError(
                f"no matching distillation strides: student={self.student_strides}, teacher={self.teacher_strides}, "
                f"requested={sorted(self.strides)}, cross={self.cross_stride_pairs}"
            )

        total = total / matched
        components = components / matched
        if imgs is not None and self.batch_has_small_target(targets, imgs):
            total = total * self.small_gain
            components[0] = components[0] * self.small_gain
        return total, components
