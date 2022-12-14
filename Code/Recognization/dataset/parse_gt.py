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


def to_gtbox(gt, rescale_fac):
    result = []
    for coor in gt:
        coor_x_list = [float(coor[2 * i]) for i in range(4)]
        coor_y_list = [float(coor[2 * i + 1]) for i in range(4)]
        label = coor[-1]
        xmin = int(min(coor_x_list))
        xmax = int(max(coor_x_list))
        ymin = int(min(coor_y_list))
        ymax = int(max(coor_y_list))

        if rescale_fac > 1.0:
            xmin = int(xmin / rescale_fac)
            xmax = int(xmax / rescale_fac)
            ymin = int(ymin / rescale_fac)
            ymax = int(ymax / rescale_fac)

        result.append((xmin, ymin, xmax, ymax, label))
    return result


def parse_gt(path, rescale_fac):
    """
    parse ICPR2018 dataset file label
    :param path: ICPR2018/txt_train
    :param rescale_fac: rescale factor
    :return: gtbox after segmentation by 16px width
    """
    return to_gtbox(read_label(path), rescale_fac)
