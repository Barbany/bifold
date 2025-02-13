import math

from einops.layers.torch import Rearrange
from torch import nn

from . import Components
from .modules import ConvDecoder, TransformerDecoder


class PickPlaceConvDecoder(nn.Module):
    def __init__(
        self,
        dim,
        is_bimanual,
        fusion_model,
        num_patches,
        patch_size,
        compute_mask=False,
        detach_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.is_bimanual = is_bimanual
        self.fusion = Components.get_by_name(fusion_model, dim=dim, **kwargs)
        self.num_patches_sqrt = int(math.sqrt(num_patches))

        if compute_mask:
            self.detach_mask = detach_mask

            self.mask_head = ConvDecoder(dim)
        else:
            self.mask_head = None

        if self.is_bimanual:
            # pick decoder
            self.left_pick_decoder = ConvDecoder(dim)
            self.right_pick_decoder = ConvDecoder(dim)

            # place decoder
            self.left_place_decoder = ConvDecoder(dim)
            self.right_place_decoder = ConvDecoder(dim)
        else:
            # pick decoder
            self.pick_decoder = ConvDecoder(dim)

            # place decoder
            self.place_decoder = ConvDecoder(dim)

    def reshape_output(self, x, input_dim):
        x = x.view(
            x.size(0),
            self.num_patches_sqrt,
            self.num_patches_sqrt,
            input_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, *inputs, **kwargs):
        return_dict = {}

        dim = inputs[-1].shape[-1]

        fused_features, return_dict["attn_weights"] = self.fusion(*inputs, **kwargs)

        if self.mask_head is not None:
            # Obviate patch token and unpatchify
            mask = self.mask_head(self.reshape_output(inputs[-1][:, 1:, :], dim))
            return_dict["mask_heatmap"] = mask.squeeze(1).sigmoid()

        fused_features = self.reshape_output(fused_features[:, 1:, :], dim)

        if self.is_bimanual:
            if self.mask_head is None:
                return_dict["left_pick_heatmap"] = (
                    self.left_pick_decoder(fused_features).squeeze(1).sigmoid()
                )
                return_dict["right_pick_heatmap"] = (
                    self.right_pick_decoder(fused_features).squeeze(1).sigmoid()
                )
            else:
                if self.detach_mask:
                    return_dict["left_pick_heatmap"] = (
                        self.left_pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"].detach()
                    )
                    return_dict["right_pick_heatmap"] = (
                        self.right_pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"].detach()
                    )
                else:
                    return_dict["left_pick_heatmap"] = (
                        self.left_pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"]
                    )
                    return_dict["right_pick_heatmap"] = (
                        self.right_pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"]
                    )

            return_dict["left_place_heatmap"] = (
                self.left_place_decoder(fused_features).squeeze(1).sigmoid()
            )
            return_dict["right_place_heatmap"] = (
                self.right_place_decoder(fused_features).squeeze(1).sigmoid()
            )

        else:
            if self.mask_head is None:
                return_dict["pick_heatmap"] = self.pick_decoder(fused_features).squeeze(1).sigmoid()
            else:
                if self.detach_mask:
                    return_dict["pick_heatmap"] = (
                        self.pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"].detach()
                    )
                else:
                    return_dict["pick_heatmap"] = (
                        self.pick_decoder(fused_features).squeeze(1).sigmoid()
                        * return_dict["mask_heatmap"]
                    )

            return_dict["place_heatmap"] = self.place_decoder(fused_features).squeeze(1).sigmoid()
        return return_dict


class PickPlaceTransDecoder(nn.Module):
    def __init__(
        self,
        dim,
        is_bimanual,
        patch_size,
        num_patches,
        fusion_model,
        compute_mask,
        condition_place_on_pick,
        detach_mask,
        **kwargs,
    ):
        super().__init__()
        self.is_bimanual = is_bimanual

        self.unpatchify = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(math.sqrt(num_patches)),
            p1=patch_size,
            p2=patch_size,
        )
        if compute_mask:
            self.detach_mask = detach_mask

            self.mask_head = TransformerDecoder(
                dim=dim,
                out_channels=1,
                num_patches=num_patches,
                patch_size=patch_size,
                **kwargs,
            )
        else:
            self.mask_head = None

        self.pick_fusion = Components.get_by_name(fusion_model, dim=dim, **kwargs)
        self.place_fusion = Components.get_by_name(fusion_model, dim=dim, **kwargs)

        out_channels = 2 if self.is_bimanual else 1
        # pick decoder
        self.pick_decoder = TransformerDecoder(
            dim=dim,
            out_channels=out_channels,
            num_patches=num_patches,
            patch_size=patch_size,
            **kwargs,
        )

        # place decoder
        self.place_decoder = TransformerDecoder(
            dim=dim,
            out_channels=out_channels,
            num_patches=num_patches,
            patch_size=patch_size,
            **kwargs,
        )

        if condition_place_on_pick:
            assert self.pick_decoder.out_dim == self.place_decoder.out_dim
            self.pick_place_fusion = Components.get_by_name(
                fusion_model, dim=self.pick_decoder.out_dim, **kwargs
            )
        else:
            self.pick_place_fusion = None

    def forward(self, *inputs, **kwargs):
        return_dict = {}

        fused_features_pick, return_dict["pick_attn_weights"] = self.pick_fusion(*inputs, **kwargs)

        fused_features_place, return_dict["place_attn_weights"] = self.place_fusion(
            *inputs, **kwargs
        )

        if self.mask_head is not None:
            # Obviate patch token and unpatchify
            mask = self.mask_head(inputs[-1])
            mask = self.unpatchify(mask).squeeze(1)
            return_dict["mask_heatmap"] = mask.sigmoid()

        pick_heatmaps = self.pick_decoder(fused_features_pick)

        place_heatmaps = self.place_decoder(fused_features_place)

        if self.pick_place_fusion is not None:
            place_heatmaps, return_dict["pick_place_attn_weights"] = self.pick_place_fusion(
                pick_heatmaps, place_heatmaps
            )

            pick_heatmaps = self.unpatchify(pick_heatmaps)
        place_heatmaps = self.unpatchify(place_heatmaps)

        if self.is_bimanual:
            left_pick_heatmap = pick_heatmaps[:, 0, :, :]
            right_pick_heatmap = pick_heatmaps[:, 1, :, :]

            if self.mask_head is None:
                return_dict["left_pick_heatmap"] = (left_pick_heatmap).sigmoid()
                return_dict["right_pick_heatmap"] = (right_pick_heatmap).sigmoid()
            else:
                if self.detach_mask:
                    return_dict["left_pick_heatmap"] = (
                        return_dict["mask_heatmap"].detach() * left_pick_heatmap.sigmoid()
                    )
                    return_dict["right_pick_heatmap"] = (
                        return_dict["mask_heatmap"].detach() * right_pick_heatmap.sigmoid()
                    )
                else:
                    return_dict["left_pick_heatmap"] = (
                        return_dict["mask_heatmap"] * left_pick_heatmap.sigmoid()
                    )
                    return_dict["right_pick_heatmap"] = (
                        return_dict["mask_heatmap"] * right_pick_heatmap.sigmoid()
                    )

            left_place_heatmap = place_heatmaps[:, 0, :, :]
            right_place_heatmap = place_heatmaps[:, 1, :, :]
            return_dict["left_place_heatmap"] = left_place_heatmap.sigmoid()
            return_dict["right_place_heatmap"] = right_place_heatmap.sigmoid()
        else:
            pick_heatmap = pick_heatmaps.squeeze(1)

            if self.mask_head is None:
                return_dict["pick_heatmap"] = pick_heatmap.sigmoid()
            else:
                if self.detach_mask:
                    return_dict["pick_heatmap"] = (
                        return_dict["mask_heatmap"].detach() * pick_heatmap.sigmoid()
                    )
                else:
                    return_dict["pick_heatmap"] = (
                        return_dict["mask_heatmap"] * pick_heatmap.sigmoid()
                    )

            place_heatmap = place_heatmaps.squeeze(1)
            return_dict["place_heatmap"] = place_heatmap.sigmoid()
        return return_dict
