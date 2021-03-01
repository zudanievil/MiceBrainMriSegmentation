import matplotlib.pyplot as plt
import skimage.transform


def plot_transform_prediction(image, coordinates, pred_h, target_h):
    p_coo = skimage.transform.matrix_transform(coordinates, pred_h)
    t_coo = skimage.transform.matrix_transform(coordinates, target_h)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()
    axs[0].imshow(image)
    axs[0].plot(coordinates[..., 0], coordinates[..., 1], c='black', marker='o', label='initial')
    axs[1].imshow(skimage.transform.warp(image, pred_h))
    axs[1].plot(p_coo[..., 0], p_coo[..., 1], c='blue', marker='o', label='prediction')
    axs[2].imshow(skimage.transform.warp(image, target_h))
    axs[2].plot(t_coo[..., 0], t_coo[..., 1], c='red', marker='o', label='ground_truth')
    fig.legend()
    return fig, axs
