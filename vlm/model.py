"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def inject_prefix(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """
        Shared helper for cls and all_patches prefix injection.
        """
        batch_size, num_visual_tokens, _ = visual_embeds.shape

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attention_mask = torch.ones(
            batch_size,
            num_visual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        stitched_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1,
        )

        stitched_labels = None
        if labels is not None:
            visual_labels = torch.full(
                (batch_size, num_visual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )

            stitched_labels = torch.cat(
                [visual_labels, labels],
                dim=1,
            )

        image_start = 0
        image_len = num_visual_tokens

        return (
            inputs_embeds,
            stitched_attention_mask,
            stitched_labels,
            image_start,
            image_len,
        )

    def inject_interleaved(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """
        Interleaved placeholder strategy.

        The prompt contains one <image> token. At runtime, that one token is
        replaced by the full visual-token sequence.

        Example:
            [Question, ..., <image>, ..., Answer]
        becomes:
            [Question, ..., CLS_visual, patch_1, ..., patch_N, ..., Answer]
        """
        if self.image_token_id is None:
            raise ValueError("image_token_id must be provided for interleaved mode.")

        batch_size, num_visual_tokens, _ = visual_embeds.shape

        stitched_embeds_list = []
        stitched_mask_list = []
        stitched_labels_list = []
        image_starts = []

        for b in range(batch_size):
            image_positions = (input_ids[b] == self.image_token_id).nonzero(
                as_tuple=False
            )

            if image_positions.numel() != 1:
                raise ValueError(
                    "Each example must contain exactly one <image> token for "
                    f"interleaved mode, but example {b} has {image_positions.numel()}."
                )

            image_pos = int(image_positions[0].item())
            image_starts.append(image_pos)

            before_embeds = text_embeds[b, :image_pos, :]
            after_embeds = text_embeds[b, image_pos + 1 :, :]

            stitched_embeds = torch.cat(
                [before_embeds, visual_embeds[b], after_embeds],
                dim=0,
            )
            stitched_embeds_list.append(stitched_embeds)

            before_mask = attention_mask[b, :image_pos]
            after_mask = attention_mask[b, image_pos + 1 :]

            visual_mask = torch.ones(
                num_visual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            stitched_mask = torch.cat(
                [before_mask, visual_mask, after_mask],
                dim=0,
            )
            stitched_mask_list.append(stitched_mask)

            if labels is not None:
                before_labels = labels[b, :image_pos]
                after_labels = labels[b, image_pos + 1 :]

                visual_labels = torch.full(
                    (num_visual_tokens,),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                stitched_labels = torch.cat(
                    [before_labels, visual_labels, after_labels],
                    dim=0,
                )
                stitched_labels_list.append(stitched_labels)

        inputs_embeds = torch.stack(stitched_embeds_list, dim=0)
        stitched_attention_mask = torch.stack(stitched_mask_list, dim=0)

        stitched_labels = None
        if labels is not None:
            stitched_labels = torch.stack(stitched_labels_list, dim=0)

        image_start = image_starts[0]
        image_len = num_visual_tokens

        return (
            inputs_embeds,
            stitched_attention_mask,
            stitched_labels,
            image_start,
            image_len,
        )

    def build_decoder_attention_mask(
        self,
        mask_mode: MaskMode,
        stitched_attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_start: int,
        image_len: int,
    ):
        if mask_mode == "causal":
            return stitched_attention_mask

        if mask_mode == "image_bidir":
            n_visual = image_len
            n_text = inputs_embeds.shape[1] - image_len

            bidir_mask = build_image_bidir_mask(
                n_visual=n_visual,
                n_text=n_text,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )  # (1, 1, T, T), additive

            # Convert padding mask to additive: (B, 1, 1, T)
            # stitched_attention_mask: 1=real token, 0=padding
            # We want: 0=attend, -inf=don't attend
            padding_mask = (1.0 - stitched_attention_mask.to(dtype=inputs_embeds.dtype))
            padding_mask = padding_mask * torch.finfo(inputs_embeds.dtype).min
            padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, T)

            # Combine: broadcasts to (B, 1, T, T)
            combined = bidir_mask + padding_mask

            return combined

        raise ValueError(f"Unknown mask mode: {mask_mode}")

    def encode_visual_tokens(
        self,
        images: torch.Tensor,
        injection: InjectionMode,
    ) -> torch.Tensor:
        """
        Encode images with the ViT and project visual features into decoder space.

        cls:
            ViT returns (B, d_image), projector returns (B, 1, d_decoder)

        all_patches / interleaved:
            ViT returns (B, N_vis, d_image), projector returns (B, N_vis, d_decoder)
        """
        if injection == "cls":
            image_features = self.vit(images)  # (B, d_image)

        elif injection in {"all_patches", "interleaved"}:
            image_features = self.vit(images, return_all_tokens=True)  # (B, N_vis, d_image)

        else:
            raise ValueError(f"Unknown injection mode: {injection}")

        visual_embeds = self.projector(image_features)
        return visual_embeds


    def inject_cls(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """
        CLS-only prefix strategy.

        Sequence becomes:
            [CLS_visual, text_1, text_2, ..., text_T]
        """
        return self.inject_prefix(
            visual_embeds=visual_embeds,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def inject_all_patches(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """
        All-patches prefix strategy.

        Sequence becomes:
            [CLS_visual, patch_1, ..., patch_N, text_1, ..., text_T]
        """
        return self.inject_prefix(
            visual_embeds=visual_embeds,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        # TODO: implement.
        # Sketch:
        #   1. Encode images with self.vit to get visual features.
        #      - "cls" -> (B, 1, d_image)
        #      - "all_patches" / "interleaved" -> (B, N+1, d_image)
        #        (you'll need to add a `return_all_tokens=True` flag to your ViT)
        #   2. Project to decoder dim with self.projector.
        #   3. Get text embeddings from the decoder's embed layer.
        #   4. Stitch visual and text tokens together according to `injection`.
        #   5. If `mask_mode == "image_bidir"`, build a custom 4D attention mask
        #      with vlm.masking.build_image_bidir_mask() and pass it to the
        #      decoder. Otherwise let the decoder use its default causal mask.
        #   6. Run the decoder with inputs_embeds=stitched, labels=adjusted_labels.
        #   7. Return {"loss": ..., "logits": ...}.
        visual_embeds = self.encode_visual_tokens(images, injection)

        text_embed_layer = self.decoder.get_input_embeddings()
        text_embeds = text_embed_layer(input_ids)

        # Match dtype with decoder embeddings, especially if decoder is bf16.
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

        if injection == "cls":
            (
                inputs_embeds,
                stitched_attention_mask,
                stitched_labels,
                image_start,
                image_len,
            ) = self.inject_cls(
                visual_embeds=visual_embeds,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        elif injection == "all_patches":
            (
                inputs_embeds,
                stitched_attention_mask,
                stitched_labels,
                image_start,
                image_len,
            ) = self.inject_all_patches(
                visual_embeds=visual_embeds,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        elif injection == "interleaved":
            (
                inputs_embeds,
                stitched_attention_mask,
                stitched_labels,
                image_start,
                image_len,
            ) = self.inject_interleaved(
                visual_embeds=visual_embeds,
                input_ids=input_ids,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        else:
            raise ValueError(f"Unknown injection mode: {injection}")

        decoder_attention_mask = self.build_decoder_attention_mask(
            mask_mode=mask_mode,
            stitched_attention_mask=stitched_attention_mask,
            inputs_embeds=inputs_embeds,
            image_start=image_start,
            image_len=image_len,
        )

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=stitched_labels,
        )

        result = {"logits": outputs.logits}

        if labels is not None:
            result["loss"] = outputs.loss

        return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Useful for §5's qualitative evaluation problem (vlm_qualitative).
        """
        self.eval()

        device = images.device

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        visual_embeds = self.encode_visual_tokens(images, injection)

        text_embed_layer = self.decoder.get_input_embeddings()
        text_embeds = text_embed_layer(input_ids)

        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

        if injection == "cls":
            (
                inputs_embeds,
                stitched_attention_mask,
                _,
                _,
                _,
            ) = self.inject_cls(
                visual_embeds=visual_embeds,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=None,
            )

        elif injection == "all_patches":
            (
                inputs_embeds,
                stitched_attention_mask,
                _,
                _,
                _,
            ) = self.inject_all_patches(
                visual_embeds=visual_embeds,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=None,
            )

        elif injection == "interleaved":
            (
                inputs_embeds,
                stitched_attention_mask,
                _,
                _,
                _,
            ) = self.inject_interleaved(
                visual_embeds=visual_embeds,
                input_ids=input_ids,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=None,
            )

        else:
            raise ValueError(f"Unknown injection mode: {injection}")

        generated_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=stitched_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return decoded