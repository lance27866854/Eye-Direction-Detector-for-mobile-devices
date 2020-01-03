def error(left_position, right_position, gt_left_position, gt_right_position):
    '''
    left_position, right_position are lists with two elements:
        The first element is the x position predicted.
        The second element is the y position predicted.
    gt_left_position, gt_right_position are still lists with two elements, and you can just directly read them form region_points.npy.
    '''
    dl = pow(pow(left_position[0]-gt_left_position[0], 2)+pow(left_position[1]-gt_left_position[1], 2), 0.5)
    dr = pow(pow(right_position[0]-gt_right_position[0], 2)+pow(right_position[1]-gt_right_position[1], 2), 0.5)
    dl_r = pow(pow(gt_left_position[0]-gt_right_position[0], 2)+pow(gt_left_position[1]-gt_right_position[1], 2), 0.5)
    Nerror = dl/dl_r if dl>dr else dr/dl_r

    return Nerror