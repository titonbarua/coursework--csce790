import numpy as np
import skimage.data
from skimage.measure import block_reduce
from scipy.signal import convolve2d
import matplotlib
import matplotlib.pyplot as plt

PREVIEW = False

# Read a cat image from skimage library and convert it to grayscale.
img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

# Create layer 1 with two predefined convolutional filters, one for detecting
# vertical lines. Another for horizontal.
l1_filter = np.zeros((2, 3, 3))
l1_filter[0, :, :] = np.array(
    [-1, 0, 1,
     -1, 0, 1,
     -1, 0, 1]).reshape(3, 3)
l1_filter[1, :, :] = np.array(
    [1, 1, 1,
     0, 0, 0,
     -1, -1, -1]).reshape(3, 3)


def relu(feature_map):
    rslt = feature_map.copy()
    neg_indices = np.argwhere(feature_map < 0.0)
    rslt[neg_indices[:, 0], neg_indices[:, 1]] = 0.0
    return rslt


def conv(img, filters):
    # print("Image shape: {}".format(img.shape))
    # print("Filters shape: {}".format(filters.shape))
    if len(filters.shape) > 3:
        return np.stack(
            [
                np.sum(
                    [
                        convolve2d(img[:, :, i], flt[:, :, i], mode='full')
                        for i in range(flt.shape[-1])
                    ],
                    axis=0
                )
                for flt in filters
            ],
            axis=2)
    else:
        return np.stack(
            [convolve2d(img, flt, mode='full') for flt in filters],
            axis=2)


def max_pooling(feature_map):
    pooled_features = []
    for feat in np.rollaxis(feature_map, 2):
        pooled_features.append(block_reduce(feat, block_size=2, func=np.max))
    return np.stack(
        pooled_features,
        axis=2)


def display_image_tile(images, titles, n_cols, savepath=None):
    n_rows = len(images) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    # plt.subplots_adjust()

    for i in range(len(images)):
        r = i // n_cols
        c = i - r * n_cols

        if n_cols == 1:
            ax = axes[r]
        elif n_rows == 1:
            ax = axes[c]
        else:
            ax = axes[r, c]

        ax.imshow(images[i])
        ax.set_title(titles[i])

    plt.tight_layout()

    if PREVIEW:
        plt.show()
    else:
        if savepath:
            print(f"Saving image {savepath} ...")
            plt.savefig(savepath, dpi=300)


l1_feature_map = conv(img, l1_filter)
# display_image_tile(
#     images=[l1_feature_map[:, :, 0], l1_feature_map[:, :, 1]],
#     titles=["L1-Map1", "L1-Map2"],
#     n_cols=2)

l1_feature_map_relu = relu(l1_feature_map)
# display_image_tile(
#     images=[l1_feature_map_relu[:, :, 0], l1_feature_map_relu[:, :, 1]],
#     titles=["L1-Map1-ReLU", "L1-Map2-ReLU"],
#     n_cols=2)

l1_feature_map_relu_pool = max_pooling(l1_feature_map_relu)
# display_image_tile(
#     images=[l1_feature_map_relu_pool[:, :, 0], l1_feature_map_relu_pool[:, :, 1]],
#     titles=["L1-Map1-ReLU-Pool", "L1-Map2-ReLU-Pool"],
#     n_cols=2)
display_image_tile(
    images=[
        l1_feature_map[:, :, 0],
        l1_feature_map[:, :, 1],
        l1_feature_map_relu[:, :, 0],
        l1_feature_map_relu[:, :, 1],
        l1_feature_map_relu_pool[:, :, 0],
        l1_feature_map_relu_pool[:, :, 1],
    ],
    titles=[
        "L2-Map1",
        "L2-Map2",
        "L2-Map1-ReLU",
        "L2-Map2-ReLU",
        "L2-Map1-ReLU-Pool",
        "L2-Map2-ReLU-Pool",
    ],
    n_cols=2,
    savepath="layer1.png"
)


# Second conv layer.
l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
l2_feature_map_relu = relu(l2_feature_map)
l2_feature_map_relu_pool = max_pooling(l2_feature_map_relu)

# print(l2_feature_map.shape)
display_image_tile(
    images=[
        l2_feature_map[:, :, 0],
        l2_feature_map[:, :, 1],
        l2_feature_map[:, :, 2],

        l2_feature_map_relu[:, :, 0],
        l2_feature_map_relu[:, :, 1],
        l2_feature_map_relu[:, :, 2],

        l2_feature_map_relu_pool[:, :, 0],
        l2_feature_map_relu_pool[:, :, 1],
        l2_feature_map_relu_pool[:, :, 2],

    ],
    titles=[
        "L2-Map1",
        "L2-Map2",
        "L2-Map3",

        "L2-Map1-ReLU",
        "L2-Map2-ReLU",
        "L2-Map3-ReLU",

        "L2-Map1-ReLU-Pool",
        "L2-Map2-ReLU-Pool",
        "L2-Map3-ReLU-Pool",
    ],
    n_cols=3,
    savepath="layer2.png"
)

# Third conv layer.
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
l3_feature_map_relu = relu(l3_feature_map)
l3_feature_map_relu_pool = max_pooling(l3_feature_map_relu)

# print(l3_filter.shape)
# print(l3_feature_map.shape)

display_image_tile(
    images=[
        l3_feature_map[:, :, 0],
        l3_feature_map_relu[:, :, 0],
        l3_feature_map_relu_pool[:, :, 0],
    ],
    titles=[
        "L3-Map1",
        "L3-Map1-ReLU",
        "L3-Map1-ReLU-Pool",
    ],
    n_cols=3,
    savepath="layer3.png"
)



