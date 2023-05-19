import numpy as np
import os


def read_file(filelocation):
    with open(filelocation, 'rb') as file:
        txt = file.read().decode('utf-16')
        
    txt_array = txt.split('\n')

    # Get index for lines with order in them
    inds_for_order = []
    for i in range(len(txt_array)):
        if 'Order' in txt_array[i]:
            inds_for_order.append(i)
    nr_of_orders = len(inds_for_order)

    # Here save order number and WL range
    order_num = []
    WL_range = []
    for order_ind in inds_for_order:
        # First split string by space
        split = txt_array[order_ind].split(' ')
        # Loop over and get the ones that contain a .
        numbers = []
        for i in range(len(split)):
            if '.' in split[i]:
                numbers.append(split[i])
        order_num.append(int(float(numbers[0])))
        WL_range.append([float(numbers[1]), float(numbers[2])])

    # Get all the data for each order
    order_data = []
    for i in range(nr_of_orders):
        if i+1 < nr_of_orders:
            points_string = txt_array[inds_for_order[i]+1:inds_for_order[i+1]]
        else:
            points_string = txt_array[inds_for_order[i]+1:]
        array_of_points = []
        for str in points_string:
            point_string_array = str.split(' ')
            point_array = []
            for point in point_string_array:
                if len(point) > 1:
                    point_array.append(float(point))
            if len(point_array) > 0:
                array_of_points.append(point_array)
        if len(array_of_points) > 0:
            order_data.append(array_of_points)

    order_WL = []
    order_points = []
    for i in range(nr_of_orders):
        WL = []
        points = []
        for j in range(len(order_data[i])):
            WL.append(order_data[i][j][0])
            points.append(order_data[i][j][1:])
        order_WL.append(WL)
        order_points.append(points)

    order_num = np.array(order_num)
    WL_range = np.array(WL_range)
    order_WL = np.array(order_WL)
    order_points = np.array(order_points)

    return order_num, WL_range, order_WL, order_points

def find_affine(nps_t):
    # translation
    nps = nps_t.copy()
    tx = -nps[0][0]
    ty = -nps[0][1]
    nps[:,0] += tx
    nps[:,1] += ty

    # rotation
    if nps[1][1] != nps[3][1]:
        angle = np.arctan(abs(nps[3][1]) / abs(nps[3][0]))
    else:
        angle = 0
    if nps[3][1] > 0:
        angle = -angle

    m = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])

    # shear
    if nps[4][0] != nps[2][0]:
        shear = (nps[4][0] - nps[2][0]) / (nps[2][1] - nps[4][1])
    else:
        shear = 0

    m = [[1, shear], [0, 1]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])

    # scale
    sx = np.abs(nps[3][0] - nps[1][0])
    sy = np.abs(nps[4][1] - nps[2][1])

    return tx, ty, angle, shear, sx, sy

def read_psfs(folder):
    files = os.listdir(folder)
    psfs = []
    wls = []
    orders = []
    for file in files:
        with open(folder+'/'+file, 'rb') as f:
            txt = f.read().decode('utf-16')

        txt_array = txt.split('\n')
        
        # get wavelength
        line_of_wl = 8
        line = txt_array[line_of_wl]
        split = line.split(' ')
        wl = float(split[0])

        # get order
        # order is in title
        split = file.split('_')
        order = int(split[2])
        
        # now read the psf:
        start_line = 21
        if len(txt_array[start_line]) == 0:
            start_line += 1
        end_line = len(txt_array)-2
        start_line = end_line-63
        psf = []
        for i in range(start_line, end_line+1):
            split = txt_array[i].split(' ')
            split = [j for j in split if j]
            split = [float(j) for j in split]
            psf.append(split)
        psf = np.array(psf)
        
        psfs.append(psf)
        wls.append(wl)
        orders.append(order)

    # Sort by orders and wls
    orders = np.array(orders)
    wls = np.array(wls)
    psfs = np.array(psfs)
    inds = np.lexsort((wls, orders))
    orders = orders[inds]
    wls = wls[inds]
    psfs = psfs[inds]

    return psfs, wls, orders

