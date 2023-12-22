import cv2
import math
import torch
import pickle
import matplotlib
import numpy as np


__all__ = [
    't2i_collate_fn', 'controlnet_collate_fn',
    'pkl_load', 'pkl_save',
    'animate_collate_fn'
]


def t2i_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def controlnet_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat([example["input_ids"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def animate_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
    reference_pixel_values = reference_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat([example["input_ids"] for example in examples])

    if 'reference_image' in examples[0]:
        reference_image = torch.cat([example["reference_image"] for example in examples])
    else:
        reference_image = torch.zeros([0])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "reference_pixel_values": reference_pixel_values,
        "input_ids": input_ids,
        "reference_image": reference_image,
    }


def pkl_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out


def draw_bodypose(canvas, kpts, kpt_valid, stickwidth=4,
                  radius=4, alpha=0.6, show_number=False):
    """
    Args:
        canvas (ndarray): shape[H, W, 3]
        kpts (ndarray): shape[n, 18, 2]
        kpt_valid (ndarray: bool): shape[n, 18]
        stickwidth (int): Default 4.
        radius (int): Default 4.
        alpha (float): Default 0.6.

    Returns: `canvas`: shape[H, W, 3]
    """
    links = [[1, 2], [1, 5],
             [2, 3], [3, 4],
             [5, 6], [6, 7],
             [1, 8], [8, 9], [9, 10],
             [1, 11], [11, 12], [12, 13],
             [1, 0], [0, 14], [14, 16],
             [0, 15], [15, 17]]

    colors = [[255, 0, 0], [255, 85, 0],
              [255, 170, 0], [255, 255, 0], [170, 255, 0],
              [85, 255, 0], [0, 255, 0], [0, 255, 85],
              [0, 255, 170], [0, 255, 255], [0, 170, 255],
              [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # draw links
    for link, color in zip(links, colors):
        for kpt, is_valid in zip(kpts, kpt_valid):
            if np.sum(is_valid[link]) == 2:
                kpt_XY = kpt[link]
                mean_XY = np.mean(kpt_XY, axis=0)
                diff_XY = kpt_XY[0] - kpt_XY[1]
                length = np.sqrt(np.square(diff_XY).sum())
                angle = math.degrees(math.atan2(diff_XY[1], diff_XY[0]))
                polygon = cv2.ellipse2Poly((int(mean_XY[0]), int(mean_XY[1])),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, [int(float(c) * alpha) for c in color])

    # draw points
    for i, color in enumerate(colors):
        for kpt, is_valid in zip(kpts, kpt_valid):
            if is_valid[i]:
                cv2.circle(
                    canvas, (int(kpt[i, 0]), int(kpt[i, 1])),
                    radius, color, thickness=-1)
                if show_number:
                    cv2.putText(
                        canvas,
                        str(i), (int(kpt[i, 0]), int(kpt[i, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0),
                        lineType=cv2.LINE_AA)
    return canvas


def draw_handpose(canvas, kpts, kpt_valid, radius=4, show_number=False):
    links = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    H, W, _ = canvas.shape
    kpts[..., 0] = np.clip(kpts[..., 0], 0, W - 1)
    kpts[..., 1] = np.clip(kpts[..., 1], 0, H - 1)
    for kpt, is_valid in zip(kpts, kpt_valid):
        kpt = kpt.astype(np.int32)
        # draw links
        for i, link in enumerate(links):
            if np.sum(is_valid[link]) == 2:
                x1, y1 = kpt[link[0]]
                x2, y2 = kpt[link[1]]
                cv2.line(
                    canvas, (x1, y1), (x2, y2),
                    matplotlib.colors.hsv_to_rgb(
                        [i / len(links), 1.0, 1.0]) * 255,
                    thickness=2)
        # draw points
        for i, (x, y) in enumerate(kpt):
            if is_valid[i]:
                cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
                if show_number:
                    cv2.putText(
                        canvas,
                        str(i), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0),
                        lineType=cv2.LINE_AA)
    return canvas
