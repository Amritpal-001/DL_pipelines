import SimpleITK as sitk


# code for automatic grid wise parcellation

hct_margins = [[0.1, 0.1], [0.12, 0.06], [0.05, 0.05]]
# hct_margins = [[0.1, 0.4], [0.12, 0.06], [0.05, 0.05]]

# hct_margins = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def threshold_ct(sitk_img):
    img = sitk.GetArrayFromImage(sitk_img)
    bone_window = np.clip((img + 1000) / 3000, 0, 1)
    thresh = bone_window > 0.2  # otsu on an image
    return thresh


def find_crop(thresh, margins=hct_margins):
    """Given thresholded image, finds crop."""
    lims = []
    for dim in range(3):
        m = thresh.mean(tuple({0, 1, 2} - {dim}))
        t = _threshold_ct_line(m)
        try:
            margin = margins[dim]
        except TypeError:
            margin = (margins, margins)

        lims.append(_longest_line(t, margin))

    return lims


def _threshold_ct_line(line):
    t = line > 0.2
    if t.sum() / t.size > 0.4:
        return t
    else:
        return(line / line.max()) > 0.2


def threshold_line(line, thresh=0.1):
    t = line >= thresh
    return t


def _longest_line(thresh, margin=(0.1, 0.1)):
    ends = np.concatenate([[1], np.diff(thresh), [1]])
    ends = np.where(ends)[0]

    lengths = np.diff(ends)
    mid_idxs = (ends[1:] + ends[:-1]) // 2
    is_pos = thresh[mid_idxs]

    largest = np.argmax(lengths * is_pos)
    left, right = ends[largest], ends[largest + 1]

    left = max(left - round(margin[0] * len(thresh)), 0)
    right = min(right + round(margin[1] * len(thresh)), len(thresh))

    return left, right


def _crop(img, crop):
    # print('HERE', imgs[0].shape)
    return img[crop[0, 0]: crop[0, 1], crop[1, 0]: crop[1, 1], crop[2, 0]: crop[2, 1]]


def get_label_map(fixed_image, size = 64):
    thresh = threshold_ct(fixed_image)
    label_map = np.zeros(thresh.shape)

    for slice_num in range(len(thresh)):
        ctr = 1
        img = thresh[slice_num,:,:]
        crop = get_extents_2d(img)
        img_crop = img[crop[0,0]:crop[0,1], crop[1,0]:crop[1,1]]
        for i in range(crop[0,0], crop[0,1], size):
            for j in range(crop[1,0], crop[1,1], size):
#                 print([slice_num, i ,j, ctr])
                label_map[slice_num, i:i + size, j: j+size] = ctr
                ctr = ctr + 1
    return label_map


def get_label_map_fixed3d(fixed_image, size = 64, brain_mid_line=245, crop_start=80):
    thresh = threshold_ct(fixed_image)
    label_map = np.zeros(thresh.shape)
    rows = int(np.ceil((crop[0,1] - crop[0,0])/size))
    columns = int(np.ceil((crop[1,1] - crop[1,0])/size))
    print(f'rows = {rows}, columns = {columns}')
    ctr = 1
    for r in range(rows):
        for c in range(columns):
            coords = get_grid_coords(r, c, rows, columns, size, brain_mid_line, crop_start)
            label_map[:, coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]] = ctr
            ctr = ctr + 1
    return label_map


def get_grid_coords(r, c, rows, columns, size, brain_mid_line, crop_start):
    return np.array((get_coords(r, rows, size, crop_start=crop_start), get_coords(c, columns, size, mid=brain_mid_line))).astype(int)
            
 
        
def get_coords(i, number, size, mid=None, crop_start=None):
    if mid is not None:
        if number%2 == 0:
            return np.array((mid + (i - number/2)*size, mid + (i - number/2 + 1)*size))
        else:
            return np.array((mid + int(i - number/2)*size + (np.sign(i - number/2)) * size/2, mid + (int(i - number/2) + 1)*size + (np.sign(i - number/2)) * size/2))
        
        
    if crop_start is not None:
        return (crop_start + i*size, crop_start + (i+1)*size)
        


if __name__ == 'main':
    label_map_3d = get_label_map_fixed3d(fixed_image,size=54, brain_mid_line=245, crop_start=80)
    sitk.WriteImage(sitk.GetImageFromArray(label_map_3d))