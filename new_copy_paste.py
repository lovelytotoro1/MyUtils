class CopyPaste(object):
    def __init__(self, iou_thr=0.1, max_num=20):
        self.max_num = max_num


    def __call__(self, images, segmentations, bboxs):
        # images n w h c
        # segmentation array n m w h 
        # bbox List n n 4
        # segmentation = np.array(segmentation)
        try:
            # select the backgroud image
            n = len(images)  # the number of images
            back_id = random.randint(0, n-1)
            image = images[back_id]
            segmentation = segmentations[back_id]
            bbox = bboxs[back_id]
            img_w, img_h = image[:2]

            all_mask = segmentation[0]
            for seg in segmentation:
                all_mask = np.logical_or(all_mask, seg)

            # random select the number of object copy to image
            copy_num = random.randint(0, self.max_num)  # 挑选目标的个数
            i,j = 0,0
            for c in range(copy_num):
                # random select the image
                img_id = random.randint(0, n)
                obj_img = images[img_id]
                obj_seg = segmentations[img_id]
                obj_box = bboxs[img_id]

                obj_num = len(obj_box)
                # random select the object
                obj_id = random.randint(0, obj_num)

                # copy paste to the image
                obj_w, obj_h = obj_box[2]-obj_box[0], obj_box[3]-obj_box[1]

                # 接着上次的结果开始遍历全图像
                for tmp_i in range(i, img_w):
                    j = 0
                    min_i = 1e8
                    for tmp_j in range(j, img_h):
                        if not all_mask[tmp_i: tmp_i+obj_w, tmp_j:tmp_j+obj_h].any():
                            all_mask[tmp_i: tmp_i+obj_w, tmp_j:tmp_j+obj_h] = obj_seg[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]]  # 更新all_mask
                            # update segmentation
                            mmask = np.zeros((img_w, img_h))
                            mmask[tmp_i: tmp_i+obj_w, tmp_j:tmp_j+obj_h] = obj_seg[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]]
                            segmentation.append(mmask)
                            # update bbox
                            bbox.append([tmp_i, tmp_j, tmp_i+obj_w, tmp_j+obj_h])
                            # update image
                            crop_obj = obj_img[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3], :] * obj_seg[obj_box[0]:obj_box[2], obj_box[1]:obj_box[3], np.newaxis]
                            crop_img = image[tmp_i: tmp_i+obj_w, tmp_j:tmp_j+obj_h, :]
                            
                            image[tmp_i: tmp_i+obj_w, tmp_j:tmp_j+obj_h, :] = np.where(crop_obj, crop_obj, crop_img)
                            
                            # move the pos of obj
                            i  = tmp_i + 1
                            j = tmp_j + obj_h + 1
                            min_i = min(min_i, obj_h)
                    if not min_i == 1e8:
                      tmp_i = min_i +1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
        return image, segmentation, bbox
