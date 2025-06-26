import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.decomposition import PCA


def save_predictions(out_folder, out_file_name, rgb=None, colormap="viridis", **kwargs):
    cm = plt.get_cmap(colormap)

    if rgb is not None:
        folder = os.path.join(out_folder, "rgb")
        os.makedirs(folder, exist_ok=True)
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(os.path.join(folder, out_file_name))
    else:
        rgb_img = None

    for k, val in kwargs.items():
        if val is not None:
            folder = os.path.join(out_folder, k)
            os.makedirs(folder, exist_ok=True)
            if "heatmap" in k or k == "depth":
                if "heatmap" in k:
                    val = val.squeeze().cpu().numpy()

                if len(val.shape) > 1:
                    heatmap = Image.fromarray((cm(val)[:, :, :3] * 255).astype(np.uint8))
                    if rgb_img is not None and "heatmap" in k:
                        blended = Image.blend(rgb_img, heatmap, alpha=0.3)
                        blended.save(os.path.join(folder, out_file_name))
                    else:
                        heatmap.save(os.path.join(folder, out_file_name))
            elif k == "particle_pos":
                np.save(
                    file=os.path.join(folder, out_file_name.replace(".png", ".npy")),
                    arr=val,
                )
            elif k == "viz":
                Image.fromarray(val).save(os.path.join(folder, out_file_name))
            elif k == "rgb_gt":
                Image.fromarray(val).save(os.path.join(folder, out_file_name))
            else:
                raise ValueError(f"Unrecognized argument {k}")


def visualize_action(sample, action):
    gt_colors = [(255, 0, 0), (0, 255, 0)]
    pred_colors = [(0, 0, 255), (0, 255, 255)]

    images = []

    if len(sample["raw_rgb"].shape) == 4:
        raw_rgb = sample["raw_rgb"]
        batched = True
    else:
        assert len(sample["raw_rgb"].shape) == 3
        raw_rgb = [sample["raw_rgb"]]
        batched = False

    for i, img in enumerate(raw_rgb):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if len(action.__dict__) == 2:
            if "pick" in sample and "place" in sample:
                img = _pick_place_viz(
                    img,
                    sample["pick"][i].cpu().numpy() if batched else sample["pick"],
                    sample["place"][i].cpu().numpy() if batched else sample["place"],
                    color=gt_colors[0],
                )
            img = _pick_place_viz(
                img,
                [action.__dict__["pick"][i]],
                [action.__dict__["place"][i]],
                color=pred_colors[0],
            )
        elif len(action.__dict__) == 4:
            for arm, gt_color, pred_color in zip(["left", "right"], gt_colors, pred_colors):
                if f"{arm}_pick" in sample:
                    pick = (
                        sample[f"{arm}_pick"][i].cpu().numpy() if batched else sample[f"{arm}_pick"]
                    )
                    place = (
                        sample[f"{arm}_place"][i].cpu().numpy()
                        if batched
                        else sample[f"{arm}_place"]
                    )
                    img = _pick_place_viz(
                        img,
                        pick,
                        place,
                        color=gt_color,
                    )
                img = _pick_place_viz(
                    img,
                    action.__dict__[f"{arm}_pick"][i],
                    action.__dict__[f"{arm}_place"][i],
                    color=pred_color,
                )
        else:
            raise ValueError(f"Invalid action {action} with {len(action.__dict__)} items")
        images.append(img)
    return images


def visualize_text_attention(attention_map, text, file_name, filter_text=True):
    if filter_text:
        text = [t for t in text if t != "</s>"]
        attention_map = attention_map[: len(text), : len(text)]

    sns.heatmap(
        attention_map,
        xticklabels=text,
        yticklabels=text,
        cmap="viridis",
    )

    # Save the figure
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


def visualize_image_attention(attention_map, image, file_name, patch_size):
    # Do nothing
    pass


def visualize_context_attention(attention_map, image, file_name, patch_size):
    # Do nothing
    pass


def visualize_attention_on_image(attention_map, image, file_name, patch_size):
    image = image.cpu().numpy().astype(np.uint8)

    # Reshape the attention map to match the image size
    height, width = image.shape[:2]
    attention_map = attention_map.reshape(height // patch_size, width // patch_size)

    # Resize the attention map to match the image size
    attention_map_resized = cv2.resize(
        attention_map, (width, height), interpolation=cv2.INTER_LINEAR
    )

    if attention_map.min() != attention_map.max():
        # Normalize the attention map for better visualization
        attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / (
            np.max(attention_map_resized) - np.min(attention_map_resized)
        )

        # Convert to uint8 format
        attention_map_resized = (attention_map_resized * 255).astype(np.uint8)

        # Create a heatmap from the attention map
        heatmap = cv2.applyColorMap(attention_map_resized, cv2.COLORMAP_JET)

        # Overlay the heatmap on the original image
        overlayed_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        # Save the overlayed image
        cv2.imwrite(file_name, overlayed_image)


def visualize_cross_attention(attention_map, inputs, sample_num, file_name, patch_size):
    has_context = "raw_rgb_context" in inputs
    text = [t[sample_num] for t in inputs["decoded_instruction"] if t[sample_num] != "</s>"]

    np.save(file_name.replace(".png", "raw.npy"), attention_map)

    text = ["[CLS]"] + text

    for text_idx, text_word in enumerate(text):
        # Start is after the text [CLS] token, the text tokens, and the image/context [CLS] token
        start = len(inputs["decoded_instruction"]) + 2

        if has_context:
            num_context_tokens = np.prod(
                [dim // patch_size for dim in inputs["raw_rgb"][sample_num].shape[:2]]
            )
            end = start + num_context_tokens
            for context_idx, mask in enumerate(inputs["context_attention_mask"][sample_num]):
                if mask == 1:
                    # There is a valid context
                    context = inputs["raw_rgb_context"][sample_num][context_idx]
                    visualize_attention_on_image(
                        attention_map[text_idx][start:end],
                        context,
                        file_name.replace(".png", f"text-{text_word}-context-{context_idx}.png"),
                        patch_size,
                    )

                start += num_context_tokens + 1
                end += num_context_tokens + 1

        else:
            end = start + np.prod(
                [dim // patch_size for dim in inputs["raw_rgb"][sample_num].shape[:2]]
            )

        visualize_attention_on_image(
            attention_map[text_idx][start:end],
            inputs["raw_rgb"][sample_num],
            file_name.replace(".png", f"text-{text_word}-image.png"),
            patch_size,
        )
    pass


def visualize_attention(
    inputs,
    raw_output,
    sample_num,
    out_folder,
    num_samples,
    patch_size,
):
    for attn_type in ["image_attention", "text_attention", "context_attention", "attn_weights"]:
        if attn_type in raw_output:
            output_dir = os.path.join(out_folder, attn_type, str(num_samples))
            os.makedirs(output_dir, exist_ok=True)

            attention_map = raw_output[attn_type]
            for layer_num in range(len(attention_map)):
                for head_num in range(len(attention_map[layer_num][sample_num])):
                    attn = attention_map[layer_num][sample_num][head_num]

                    if isinstance(attn, torch.Tensor):
                        attn = attn.cpu().numpy()

                    file_name = os.path.join(output_dir, f"layer-{layer_num}_head-{head_num}.png")

                    if attn_type == "text_attention":
                        visualize_text_attention(
                            attn,
                            text=[t[sample_num] for t in inputs["decoded_instruction"]],
                            file_name=file_name,
                        )
                    elif attn_type == "image_attention":
                        visualize_image_attention(
                            attn,
                            inputs["raw_rgb"][sample_num],
                            file_name,
                            patch_size,
                        )
                    elif attn_type == "context_attention":
                        visualize_context_attention(
                            attn,
                            inputs["raw_rgb_context"][sample_num],
                            file_name,
                            patch_size,
                        )
                    elif attn_type == "attn_weights":
                        visualize_cross_attention(
                            attn,
                            inputs,
                            sample_num,
                            file_name,
                            patch_size,
                        )
                    else:
                        raise NotImplementedError


def visualize_features(
    inputs,
    raw_output,
    sample_num,
    out_folder,
    num_samples,
    patch_size,
    threshold=0,
):
    patch_h = inputs["raw_rgb"][sample_num].shape[0] // patch_size
    patch_w = inputs["raw_rgb"][sample_num].shape[1] // patch_size

    num_rgb_tokens = patch_h * patch_w

    for feature_type in ["features", "out_features"]:
        if feature_type == "out_features" and "out_features" not in raw_output:
            continue

        output_dir = os.path.join(out_folder, feature_type, str(num_samples))
        os.makedirs(output_dir, exist_ok=True)

        for include_text in [True, False]:
            total_images = 1

            text = [t[sample_num] for t in inputs["decoded_instruction"] if t[sample_num] != "</s>"]

            if feature_type == "features":
                rgb_features = raw_output["image_features"][sample_num][1:]
                text_features = raw_output["text_features"][sample_num][1 : len(text) + 1]
            else:
                rgb_features = raw_output["out_features"][sample_num][-num_rgb_tokens:]
                text_features = raw_output["out_features"][sample_num][1 : len(text) + 1]

            all_features = [rgb_features]

            if "image_context_features" in raw_output:
                if feature_type == "features":
                    start = 1
                else:
                    start = 2 + len(text)
                end = start + num_rgb_tokens
                for mask in inputs["context_attention_mask"][sample_num]:
                    if mask == 1:
                        all_features.append(raw_output["image_context_features"][sample_num])
                        total_images += 1
                    start += num_rgb_tokens + 1
                    end += num_rgb_tokens + 1

            if include_text:
                all_features.append(text_features)

            all_features = torch.cat(all_features, dim=0).cpu().numpy()

            pca = PCA(n_components=3)
            pca.fit(all_features)
            pca_features = pca.transform(all_features)

            if not include_text:
                # Save the unaltered features in case threshold is not right
                raw_outputs = os.path.join(output_dir, "raw")
                os.makedirs(raw_outputs, exist_ok=True)

                # visualize PCA components for finding a proper threshold
                plt.subplot(1, 3, 1)
                plt.hist(pca_features[:, 0])
                plt.subplot(1, 3, 2)
                plt.hist(pca_features[:, 1])
                plt.subplot(1, 3, 3)
                plt.hist(pca_features[:, 2])
                plt.savefig(os.path.join(raw_outputs, "pca_components.png"))
                plt.close()

                np.save(
                    os.path.join(raw_outputs, "all_features.npy"),
                    all_features,
                )

                np.save(
                    os.path.join(raw_outputs, "pca_features.npy"),
                    pca_features,
                )

                Image.fromarray(inputs["raw_rgb"][sample_num].cpu().numpy()).save(
                    os.path.join(raw_outputs, "rgb.png")
                )

                if "image_context_features" in raw_output:
                    for mask_idx, mask in enumerate(inputs["context_attention_mask"][sample_num]):
                        if mask == 1:
                            Image.fromarray(
                                inputs["raw_rgb_context"][sample_num][mask_idx]
                                .cpu()
                                .numpy()
                                .astype(np.uint8)
                            ).save(os.path.join(raw_outputs, f"context-{mask_idx}.png"))

                # segment using the first component
                pca_features_fg = pca_features[:, 0] > threshold
            else:
                # Take everything
                pca_features_fg = np.ones(pca_features.shape[0], dtype=bool)

            # PCA for only foreground patches
            pca.fit(all_features[pca_features_fg])
            pca_features_rem = pca.transform(all_features[pca_features_fg])
            for i in range(3):
                pca_features_rem[:, i] = (
                    pca_features_rem[:, i] - pca_features_rem[:, i].mean()
                ) / (pca_features_rem[:, i].std() ** 2) + 0.5

            pca_features[~pca_features_fg] = 0
            pca_features[pca_features_fg] = pca_features_rem

            start = 0
            end = start + num_rgb_tokens

            fig, axs = plt.subplots(
                nrows=2,
                ncols=total_images + 1 if include_text else total_images,
                figsize=(16, 16),
                tight_layout=True,
            )

            ax_pca_img = axs[0, 0] if total_images > 1 or include_text else axs[0]
            ax_raw_img = axs[1, 0] if total_images > 1 or include_text else axs[1]

            ax_pca_img.imshow(pca_features[start:end].reshape(patch_h, patch_w, 3)[..., ::-1])
            ax_pca_img.set_title("PCA RGB")
            ax_pca_img.axis("off")

            ax_raw_img.imshow(inputs["raw_rgb"][sample_num].cpu().numpy().astype(np.uint8))
            ax_raw_img.set_title("RGB")
            ax_raw_img.axis("off")

            if total_images > 1:
                start += num_rgb_tokens
                end += num_rgb_tokens
                for i in range(1, total_images):
                    axs[0, i].imshow(
                        pca_features[start:end].reshape(patch_h, patch_w, 3)[..., ::-1]
                    )
                    axs[0, i].set_title(f"PCA Context-{i}")
                    axs[0, i].axis("off")

                    axs[1, i].imshow(
                        inputs["raw_rgb_context"][sample_num][i - 1].cpu().numpy().astype(np.uint8)
                    )
                    axs[1, i].set_title(f"Context-{i}")
                    axs[1, i].axis("off")
                    start += num_rgb_tokens
                    end += num_rgb_tokens

            if include_text:
                axs[0, -1].imshow(pca_features[start:].reshape(1, -1, 3))
                for i, text_i in enumerate(text):
                    axs[0, -1].annotate(
                        text_i,
                        xy=(0, i),
                        ha="center",
                        va="center",
                        color="white",
                    )
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir,
                    (
                        f"sample-{sample_num}.png"
                        if not include_text
                        else f"sample-{sample_num}-text.png"
                    ),
                ),
                dpi=300,
            )


def _pick_place_viz(img, picks, places, color):
    if not isinstance(picks, list) and len(picks.shape) == 1:
        picks = [picks]
        places = [places]
    for pick, place in zip(picks, places):
        if pick[0] >= 0:
            cv2.circle(
                img=img,
                center=(round(pick[0]), round(pick[1])),
                radius=3,
                color=color,
                thickness=2,
            )
        if place[0] >= 0:
            cv2.arrowedLine(
                img=img,
                pt1=(round(pick[0]), round(pick[1])),
                pt2=(round(place[0]), round(place[1])),
                color=color,
                thickness=2,
            )
    return img
