# Copyright (c) wilson.xu. All rights reserved.
import cv2
import math
import numpy as np
from PIL import Image

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from diffusers.data.face_train.YOLOv8FaceDet import YOLOv8FaceDet


__all__ = ['FaceProcess', 'resize_img']


IMAGE_EXTS = ['.jpg', '.jpeg', '.png']


def resize_img(img, size=1024, mode="long_side"):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    w, h = img.size
    new_size = max(w, h) if mode == "long_side" else min(w, h)
    ratio = size / new_size
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return img.resize((new_w, new_h), 1)


def pad_to_square(im):
    h, w, _ = im.shape
    ns = int(max(h, w) * 1.5)
    im = cv2.copyMakeBorder(im, int((ns - h) / 2), (ns - h) - int((ns - h) / 2), int((ns - w) / 2),
                            (ns - w) - int((ns - w) / 2), cv2.BORDER_CONSTANT, 255)
    return im


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    if s1 < 1.0e-4:
        s1 = 1.0e-4
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def face_rotate(im, keypoints):
    h, w, _ = im.shape
    dst_mean_face_size = 160
    dst_mean_face = np.asarray([0.31074522411511746, 0.2798131190011913,
                                0.6892073313037804, 0.2797830232679366,
                                0.49997367716346774, 0.5099309118810921,
                                0.35811903020866753, 0.7233174007629063,
                                0.6418878095835022, 0.7232890570786875])
    dst_mean_face = np.reshape(dst_mean_face, (5, 2)) * dst_mean_face_size

    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in keypoints]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in dst_mean_face]))
    trans_mat = transformation_from_points(pts1, pts2)
    if trans_mat[1, 1] > 1.0e-4:
        angle = math.atan(trans_mat[1, 0] / trans_mat[1, 1])
    else:
        angle = math.atan(trans_mat[0, 1] / trans_mat[0, 2])
    im = pad_to_square(im)
    ns = int(1.5 * max(h, w))
    M = cv2.getRotationMatrix2D((ns / 2, ns / 2), angle=-angle / np.pi * 180, scale=1.0)
    im = cv2.warpAffine(im, M=M, dsize=(ns, ns), flags=cv2.INTER_LANCZOS4)
    return im


def crop_image_from_face_bbox(im, bbox):
    h, w, _ = im.shape
    thre = 0.35 / 1.15
    maxf = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    cx = (bbox[2] + bbox[0]) / 2
    cy = (bbox[3] + bbox[1]) / 2
    lenp = int(maxf / thre)
    yc = 0.5 / 1.15
    xc = 0.5
    xmin = int(cx - xc * lenp)
    xmax = xmin + lenp
    ymin = int(cy - yc * lenp)
    ymax = ymin + lenp
    x1 = 0
    x2 = lenp
    y1 = 0
    y2 = lenp
    if xmin < 0:
        x1 = -xmin
        xmin = 0
    if xmax > w:
        x2 = w - (xmax - lenp)
        xmax = w
    if ymin < 0:
        y1 = -ymin
        ymin = 0
    if ymax > h:
        y2 = h - (ymax - lenp)
        ymax = h
    imc = np.zeros_like(im, shape=(lenp, lenp, 3))
    imc[y1:y2, x1:x2, :] = im[ymin:ymax, xmin:xmax, :]
    return imc


def pad_bbox_to_square(bbox, img_shape):
    x1, y1, x2, y2 = bbox
    H, W = img_shape

    w, h = x2 - x1, y2 - y1
    if w < h:
        lenp = (h - w) // 2
        x1 -= lenp
        x2 += (h - w - lenp)
        if x1 < 0:
            x2 += abs(x1)
            x1 = 0
        elif x2 > W:
            x1 -= (x2 - W)
            x2 = W
    elif h < w:
        lenp = (w - h) // 2
        y1 -= lenp
        y2 += (w - h - lenp)
        if y1 < 0:
            y2 += abs(y1)
            y1 = 0
        elif y2 > H:
            y1 -= (y2 - H)
            y2 = W
    return [x1, y1, x2, y2]


def get_square_from_mask(mask):
    x = np.where(np.sum(mask, axis=0))[0]
    y = np.where(np.sum(mask, axis=1))[0]
    if len(x) > 1 and len(y) > 1:
        x1, y1, x2, y2 = int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1
        return pad_bbox_to_square([x1, y1, x2, y2], mask.shape)
    else:
        return None


def get_mask_head(result, img_shape):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    mask_hair = np.zeros(img_shape)
    mask_face = np.zeros(img_shape)
    mask_human = np.zeros(img_shape)
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    mask_head = np.clip(mask_hair + mask_face, 0, 1)
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    mask_head = np.zeros(img_shape).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    return mask_head


class FaceProcess(object):
    def __init__(self,
                 yolov8_path="yolov8n-face.onnx",
                 human_parsing_path="damo/cv_resnet101_image-multiple-human-parsing",
                 face_landmark_path="damo/cv_manual_facial-landmark-confidence_flcm",
                 face_rotate=False,
                 size=512,
                 face_thr=0.855):
        self.face_rotate = face_rotate
        self.size = size
        self.face_thr = face_thr
        self.face_detection = YOLOv8FaceDet(yolov8_path)
        self.segmentation_pipeline = pipeline(
            Tasks.image_segmentation,
            human_parsing_path,
            model_revision='v1.0.1')
        self.facial_landmark_confidence_func = pipeline(
            Tasks.face_2d_keypoints,
            face_landmark_path,
            model_revision='v2.5')

    def get_max_bbox(self, bboxes):
        if len(bboxes) > 0:
            area = bboxes[:, 2] * bboxes[:, 3]
            ind = np.argmax(area)
            x1, y1, w, h = bboxes[ind]
            return [int(x1), int(y1), int(x1 + w), int(y1 + h)]
        else:
            return None

    def __call__(self, img):
        if isinstance(img, (Image.Image, np.ndarray)):
            img = np.array(img)
        else:
            raise Exception(f"Unsupported image type: {type(img)}.")
        # 人脸检测
        det_bboxes = self.face_detection(img)[0]
        bbox = self.get_max_bbox(det_bboxes)
        if bbox is None:
            print(f"No face detected...")
            return None

        # 高分辨率图像为了保真，先将人脸区域裁剪出来
        img = crop_image_from_face_bbox(img, bbox)

        if self.face_rotate:
            # 人脸转正
            _, _, _, landmarks = self.face_detection(img)
            img = face_rotate(img, landmarks[0, :, :2])
            det_bboxes, det_conf, det_classid, landmarks = self.face_detection(img)
            bbox = self.get_max_bbox(det_bboxes)
            if bbox is None:
                print(f"After rotate, no face detected...")
                return None
            img = crop_image_from_face_bbox(img, bbox)

        # human parsing
        seg_shape = (int(self.size * 2), int(self.size * 2))
        img = Image.fromarray(img).resize(seg_shape, 1)
        result = self.segmentation_pipeline(img)
        mask_head = get_mask_head(result, seg_shape)
        bbox = get_square_from_mask(mask_head)
        if bbox is None:
            print('parsing quality fail...')
            return None
        x1, y1, x2, y2 = bbox
        img = np.array(img)
        img = img[y1:y2, x1:x2]
        img = Image.fromarray(img).resize(
            (self.size, self.size), 1)
        # 人脸过滤: 过滤掉人脸质量差的图像
        raw_result = self.facial_landmark_confidence_func(img)
        if raw_result is None or float(raw_result['scores'][0]) < self.face_thr:
            print('landmark quality fail...')
            return None
        return img
