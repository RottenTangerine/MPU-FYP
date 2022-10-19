import matplotlib.pyplot as plt
import numpy as np

def read_label(path):
    """
    convert label to coordinate
    :param path: label path
    :return: coordinates
    """
    result = []
    with open(path, encoding='UTF-8') as f:
        for line in f.read().split('\n'):
            if line:
                data = line.split(',')
                result.append(data)
    return result


def to_gtbox(coordinates, rescale_fac):
    gt_boxes = []
    for coor in coordinates:
        coor_x_list = [float(coor[2 * i]) for i in range(4)]
        coor_y_list = [float(coor[2 * i + 1]) for i in range(4)]
        xmin = int(min(coor_x_list))
        xmax = int(max(coor_x_list))
        ymin = int(min(coor_y_list))
        ymax = int(max(coor_y_list))

        if rescale_fac > 1.0:
            xmin = int(xmin / rescale_fac)
            xmax = int(xmax / rescale_fac)
            ymin = int(ymin / rescale_fac)
            ymax = int(ymax / rescale_fac)

        _prev = xmin
        for i in range(xmin // 16 + 1, xmax // 16 + 1):
            _next = i * 16
            gt_boxes.append((_prev, ymin, _next, ymax))
            _prev = _next
        gt_boxes.append((_prev, ymin, xmax, ymax))
    return np.asarray(gt_boxes)


def parse_gt(path, rescale_fac):
    """
    parse ICPR2018 dataset file label
    :param path: ICPR2018/txt_train
    :param rescale_fac: rescale factor
    :return: gtbox after segmentation by 16px width
    """
    return to_gtbox(read_label(path), rescale_fac)