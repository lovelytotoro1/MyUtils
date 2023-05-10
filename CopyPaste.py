import os
import cv2
import numpy as np
import random
import imgaug as ia
from imgaug import augmenters as iaa

class Copy_Paste(object):
    def __init__(self, iou_thresh=0.1, epochs=30, max_copy_num=10):
        self.iou_thresh = iou_thresh
        self.epochs = epochs
        self.max_copy_num = max_copy_num

    def getBox(self, mask):
        """
            用于获取图像的外接矩形
            mask: np.array  (w, h, c)

            return:
                左上右下 x1, y2, x2, y2
        """
        if len(mask.shape) == 3:
            mask = mask.squeeze()

        y_coords, x_coords = np.nonzero(mask)
        x_min = x_coords.min()
        x_max = x_coords.max()

        y_min = y_coords.min()
        y_max = y_coords.max()

        return [x_min, y_min, x_max, y_max]

    def applyRotationAndScale(self, mask, cx, cy):
        """
            对模板图像绕cx,cy点进行随机旋转
            返回旋转之后的 box
        """
        w,h = mask.shape[:2]
        angle = random.randint(0, 360)
        scale = random.random() * 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        mask = cv2.warpAffine(mask, M, (w, h))
        return mask, angle, scale

    def computeMaxIOU(self, bboxs, box):
        mx_iou = 0
        for bbox in bboxs:
            intersection = (min(bbox[2], box[2]) - max(bbox[0], box[0])) * (min(bbox[3], box[3]) - max(bbox[1], box[1]))
            union = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (box[2] - box[0]) * (box[3] - box[1]) - intersection
            iou = intersection / union
            if mx_iou < iou:
                mx_iou = iou
        return mx_iou

    def applyCopyPaste(self, res_mask, bboxs, mask, box):
        """
            根据 bbox 将图像mask随机置于res_mask上

            res_mask: np.array  (w,h) 整张图像
            mask: np.array (w,h) 模板图像
            bboxs: 已经有的目标物体
            bbox: 当前目标物体

            return:
            res_mask: 粘贴之后的全图
            bboxs: 当前所有的目标位置

        """
        W, H = res_mask.shape[:2] # 整张图像的长宽
        flag = False  # 是否可以放置的标志
        for i in range(self.epochs):  # 随机放置位置self.epochs 次
            w, h = [box[2] - box[0], box[3] - box[1]]  # 取目标的长宽
            cx, cy = (w+1) // 2, (h+1) //2
            pos_cx, pos_cy = random.randint(cx, W-cx), random.randint(cy, H-cy)  # 取随机点为目标中心点
            new_box = [pos_cx-cx , pos_cy-cy, pos_cx-cx+w, pos_cy-cy+h]  # 得到新的box
            mx_iou = self.computeMaxIOU(bboxs, new_box)
            if mx_iou < self.iou_thresh:  # 说明这个位置合适
                res_mask[new_box[0]:new_box[2], new_box[1]:new_box[3]] = mask[box[0]:box[2], box[1]:box[3]]  # 图像粘贴
                bboxs.append(new_box)
                flag = True
                break
        return res_mask, bboxs, flag
            

    def __call__(self, temp_images: list, img_size=(5000, 10000)):
        """
            temp_images: 所有的模板图像 外面是List类型 表示模板的个数 里面是 (w * h) 的np.array类型
            img_size: copy_paste之后的图像大小
        """
        n = len(temp_images)
        bboxs = []
        for i in range(n):
            box = self.getBox(temp_images[i])
            bboxs.append(box)
        
        # 随机copy模板的个数
        copy_object_num = self.max_copy_num

        # 在 0~n-1 之间随机取copy_object_num个数
        random_list =  [random.randint(0, n-1) for _ in range(copy_object_num)]

        res_mask = np.zeros(img_size)  # 创建大图 用于返回
        res_angles = []  # 旋转角
        res_scales = []  # 尺度变化
        res_offset = []  # 偏移量
        res_bboxs = []  # 目标框的位置

        # 遍历随机copy的目标列表
        for ind in random_list:
            # 取出需要 copy_paste 的对象
            img = temp_images[ind]
            box = bboxs[ind]  # 左上右下
            cx, cy = [int((box[0] + box[2])/2), int((box[1] + box[3]) / 2)]  # 中心点坐标
            w, h = [box[2] - box[0], box[3] - box[1]]  # 长宽
            img, angle, scale = self.applyRotationAndScale(img, cx, cy)  # 对图像按照目标中心点进行旋转、缩放
            res_mask, res_bboxs, flag = self.applyCopyPaste(res_mask, res_bboxs, img, box)
            if flag:  # 说明可以添加
                added_bbox = res_bboxs[-1]  # 新添加的bbox
                offset = [(added_bbox[0]+added_bbox[2]) // 2 - (box[0] + box[2]) // 2, (added_bbox[1]+added_bbox[3]) // 2 - (box[1] + box[3]) // 2] # 中心点坐标的偏移量
                res_angles.append(angle)    # 角度
                res_scales.append(scale)    # 尺度
                res_offset.append(offset)
        return res_mask, res_angles, res_scales, res_offset, res_bboxs

if __name__ == "__main__":
    root = r'.'
    temp_imgs = []
    for i in os.listdir(root):
        template_path = os.path.join(root, i)
        img = cv2.imread(template_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(src = img, thresh=177, maxval=1, type=cv2.THRESH_BINARY)
        temp_imgs.append(np.array(img).squeeze())
    copy_paste = Copy_Paste(0.1, 30)
    
    res = copy_paste(temp_imgs)
