import torch
import torchvision


def plot_prediction_switch(
    imgs: torch.Tensor,
    labels: torch.Tensor,
    per_imgs: torch.Tensor,
    per_labels: torch.Tensor,
) -> None:
    """
    Plotting prediction switch in 5 times 5 cluster.

    :param imgs: The original images to display.
    :param labels: The original labels to display.
    :param per_imgs: The perturbated images to display.
    :param per_labels: The perturbated images to display.

    :return: None
    """
    n = 25
    col = 5
    labels = labels[:n]
    per_labels = per_labels[:n]
    per_imgs = torchvision.utils.make_grid(per_imgs[:n].detach().cpu(), col)
    imgs = torchvision.utils.make_grid(imgs[:n].detach().cpu(), col)

    if not isinstance(per_imgs, list):
        per_imgs = [per_imgs]

    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, ncols=3 * len(imgs), figsize=(14, 4))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        ax1.imshow(np.asarray(img))
        ax1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax1.set_title("Original Images")

    for i, img in enumerate(per_imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        ax2.imshow(img)
        ax2.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax2.set_title("Perturbated Images")

    col = 0
    row = 0
    for idx, (l, per_l) in enumerate(zip(labels, per_labels)):
        l, per_l = int(l.item()), int(per_l.item())
        ax3.text(0.02 + 0.2 * row, 0.88 - 0.2 * col, f"{l} → {per_l}", fontsize=13)
        row += 1
        if row % 5 == 0 and row > 0:
            col += 1
            row = 0

    for x, y in zip(ax3.xaxis.get_major_ticks(), ax3.yaxis.get_major_ticks()):
        x.tick1line.set_visible(False)
        x.tick2line.set_visible(False)
        x.label1.set_visible(False)
        x.label2.set_visible(False)
        y.tick1line.set_visible(False)
        y.tick2line.set_visible(False)
        y.label1.set_visible(False)
        y.label2.set_visible(False)

    # ax3.axis('off')
    ax3.grid(True)
    ax3.set_title("Prediction Switch")
