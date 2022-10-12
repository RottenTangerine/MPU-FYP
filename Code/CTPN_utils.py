import matplotlib.pyplot as plt
import numpy as np

def draw_gt_boxes(img, gt_boxes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)
    for box in gt_boxes:
        _p = (box[0], box[1])
        _w = box[2] - box[0]
        _h = box[3] - box[1]
        rect = plt.Rectangle(_p, _w, _h, color='red', alpha=0.2)
        ax.add_patch(rect)
    plt.show()


def gen_basic_anchor(img_size, scale):
    """
    :return:
    """
    # 10 types of basic box size
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchors = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchors[0] + base_anchors[2]) * 0.5
    yt = (base_anchors[1] + base_anchors[3]) * 0.5
    # ic(xt, yt)

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchors = np.hstack((x1, y1, x2, y2))
    h, w = img_size
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchors + [j, i, j, i])
    # ic(np.array(anchor).shape)
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    calculate two boxes IOU
    :param box1: [,x1,y1,x2,y2]  anchor
    :param boxes2: [Msample,x1,y1,x2,y2]  grouth-box
    :return:
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    :param boxes1: [Nsample,x1,y1,x2,y2]  anchor
    :param boxes2: [Msample,x1,y1,x2,y2]  grouth-box
    :return:
    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps

def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def cal_rpn(args, imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_basic_anchor(featuresize, scale)

    # calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > args.IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps < args.IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > args.RPN_POSITIVE_NUM:
        labels[np.random.choice(fg_index, len(fg_index) - args.RPN_POSITIVE_NUM, replace=False)] = -1

    # subsample negative labels
    bg_index = np.where(labels == 0)[0]
    num_bg = args.RPN_TOTAL_NUM - np.sum(labels == 1)
    if len(bg_index) > num_bg:
        # print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # calculate bbox targets
    # debug here
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])
    # bbox_targets=[]

    return [labels, bbox_targets], base_anchor
