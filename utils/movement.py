import time


def chk_movement_line_one(cent, a1, b1, p_id, cam_id, count_type, buff_dict):
    Dir = ((cent[0] - a1[0]) * (b1[1] - a1[1])) - ((cent[1] - a1[1]) * (b1[0] - a1[0]))

    Dir = 1 if Dir > 0 else -1
    flag = False

    if cam_id in buff_dict:
        if p_id in buff_dict[cam_id]:
            out = buff_dict[cam_id][p_id]
            if int(time.time()) - out['timestamp'] > 900:
                buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
            else:
                if out['d'] != Dir:
                    buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
                    flag = True
                else:
                    buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': out['d']}
        else:
            buff_dict[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
    else:
        buff_dict[cam_id] = dict()

    if flag:
        if Dir == count_type:
            return 1
        else:
            return -1
    else:
        return 0


def chk_movement_line_two(cent, a1, b1, p_id, cam_id, count_type, buff_dict_2):
    Dir = ((cent[0] - a1[0]) * (b1[1] - a1[1])) - ((cent[1] - a1[1]) * (b1[0] - a1[0]))

    Dir = 1 if Dir > 0 else -1
    flag = False

    if cam_id in buff_dict_2:
        if p_id in buff_dict_2[cam_id]:
            out = buff_dict_2[cam_id][p_id]
            if int(time.time()) - out['timestamp'] > 900:
                buff_dict_2[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
            else:
                if out['d'] != Dir:
                    buff_dict_2[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
                    flag = True
                else:
                    buff_dict_2[cam_id][p_id] = {'timestamp': int(time.time()), 'd': out['d']}
        else:
            buff_dict_2[cam_id][p_id] = {'timestamp': int(time.time()), 'd': Dir}
    else:
        buff_dict_2[cam_id] = dict()

    if flag:
        if Dir == count_type:
            return 1
        else:
            return -1
    else:
        return 0
