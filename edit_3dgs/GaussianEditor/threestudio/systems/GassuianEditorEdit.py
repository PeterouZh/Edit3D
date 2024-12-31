import shutil
import cv2
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import threestudio
import os
from scipy.ndimage import filters
from threestudio.utils.clip_metrics import ClipSimilarity

from threestudio.utils.misc import get_device
from threestudio.systems.GassuianEditor import GaussianEditor
from gaussian_editor.text_segmentation_cli import TextSegmentation


@threestudio.register("gsedit-system-edit")
class GaussianEditor_Edit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            self.update_mask()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]

        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                ):
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )

                    self.edit_frames[cur_index] = result["edit_images"].detach().clone()
                    # print("edited image index", cur_index)

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target, total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))



@threestudio.register("gsedit-system-edit-adss")
class GaussianEditor_Edit_ADSS(GaussianEditor_Edit):

    def _save_masked_images(self,
                            img_np,
                            mask_np,
                            idx,
                            color):
        img_original = img_np.copy() / 255.
        # img_colored_mask[mask_np.bool()] = np.random.random(3)
        img_colored_mask = img_original.copy()
        img_colored_mask = (1 - mask_np[..., np.newaxis]) * img_colored_mask + mask_np[..., np.newaxis] * color

        alpha = 0.9
        image_anns = np.clip((1 - alpha) * img_original + alpha * img_colored_mask, 0, 1.)

        img_anns = Image.fromarray(np.clip(image_anns * 255, 0, 255).astype(np.uint8))

        saved_images_masked = f"{self.cfg.adss.saved_dir}/images_masked"
        os.makedirs(saved_images_masked, exist_ok=True)
        img_anns.save(f"{saved_images_masked}/{idx:05d}.jpg")
        saved_masks = f"{self.cfg.adss.saved_dir}/masks"
        os.makedirs(saved_masks, exist_ok=True)
        Image.fromarray((mask_np * 255).astype(np.uint8)).save(f"{saved_masks}/{idx:05d}.jpg")
        pass

    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")

            self.masks_np = {}
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )

                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

                # mask = self.text_segmentor(
                #     self.origin_frames[id],
                #     self.cfg.seg_prompt)[0].to(get_device())

                rendered_image_pil = Image.fromarray(self.origin_frames[id][0].mul(255).byte().cpu().numpy())
                with torch.set_grad_enabled(True):
                    mask = self.text_seg.text_seg_multi_queries(
                        img_pil=rendered_image_pil,
                        query=self.cfg.adss.ts_query).astype(np.float32)
                self.masks_np[id] = mask

                if 'TL_DEBUG' in os.environ:
                    Image.fromarray((mask * 255).astype(np.uint8)).save("/home/ps/Downloads/1_mask_image.png")

                mask = torch.from_numpy(mask).to(get_device()).unsqueeze(0)

                if id in self.failed_seg_idx_list:
                    mask = torch.ones_like(mask)

                mask_to_save = (
                      mask[0]
                      .cpu()
                      .detach()[..., None]
                      .repeat(1, 1, 3)
                      .numpy()
                      .clip(0.0, 1.0)
                      * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = self.origin_frames[id].detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                      masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

                if 'TL_DEBUG' in os.environ:
                    break

            weights /= weights_cnt + 1e-7

            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            torch.save(selected_mask, gs_mask_path)
        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)

        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

    def on_fit_start(self) -> None:
        # TextSegmentation
        self.text_seg = TextSegmentation(img_size=self.cfg.adss.ts_img_size,
                                         caption=self.cfg.adss.ts_caption,
                                         blip_type=self.cfg.adss.ts_blip_type,
                                         layer_num=self.cfg.adss.ts_layer_num,
                                         blur=self.cfg.adss.ts_blur,
                                         thresh=self.cfg.adss.ts_thresh,
                                         model_type=self.cfg.adss.ts_model_type,
                                         sam_checkpoint=self.cfg.adss.ts_sam_checkpoint,
                                         use_default_sam=self.cfg.adss.ts_use_default_sam,
                                         points_per_side=self.cfg.adss.ts_points_per_side,
                                         pred_iou_thresh=self.cfg.adss.ts_pred_iou_thresh,
                                         stability_score_thresh=self.cfg.adss.ts_stability_score_thresh,
                                         crop_n_layers=self.cfg.adss.ts_crop_n_layers,
                                         crop_n_points_downscale_factor=self.cfg.adss.ts_crop_n_points_downscale_factor,
                                         min_mask_region_area=self.cfg.adss.ts_min_mask_region_area,
                                         SAM_masks_filter_thresh=self.cfg.adss.ts_SAM_masks_filter_thresh,
                                         min_area=self.cfg.adss.ts_min_area,
                                         use_mobilesam=self.cfg.adss.ts_use_mobilesam,
                                         mobilesam_model_type=self.cfg.adss.ts_mobilesam_model_type,
                                         mobilesam_ckpt=self.cfg.adss.ts_mobilesam_ckpt,
                                         device=torch.device('cuda'))

        # self.is_initial_target_image_replaced = False

        erode_kernel_size = self.cfg.adss.erode_kernel_size
        self.erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel_size = self.cfg.adss.dilate_kernel_size
        self.dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

        self.failed_seg_idx_list = list(map(int, self.cfg.adss.failed_seg_idx_list.split(','))) \
            if self.cfg.adss.failed_seg_idx_list else []

        super().on_fit_start()


        # self.origin_frames
        rand_color = np.array([1., 0, 0])
        self.masks = {}

        # for idx, image in self.origin_frames.items():
        #     self.masks[idx] = torch.zeros(image.shape[1:-1])

        for idx, original_image in tqdm(self.origin_frames.items(), desc="ADSS get masks"):

            rendered_image_pil = Image.fromarray(original_image[0].mul(255).byte().cpu().numpy())

            # mask = self.text_seg.text_seg_multi_queries(
            #     img_pil=rendered_image_pil,
            #     query=self.cfg.adss.ts_query).astype(np.float32)

            mask = self.masks_np[idx]
            # if 'TL_DEBUG' in os.environ:
            #     Image.fromarray((mask * 255).astype(np.uint8)).save("/home/ps/Downloads/1_mask_image.png")


            if self.cfg.adss.erode_mask:
                mask = cv2.erode((mask * 255).astype(np.uint8), self.erode_kernel)
                mask = mask / 255.
            if self.cfg.adss.dilate_mask:
                mask = cv2.dilate((mask * 255).astype(np.uint8), self.dilate_kernel, iterations=1)
                mask = mask / 255.

            if self.cfg.adss.smooth_mask:
                mask = filters.gaussian_filter(mask, self.cfg.adss.gaussian_filter_ratio * max(mask.shape[:2]))

            if self.cfg.adss.mask_flip:
                mask = 1. - mask

            mask_tensor = torch.from_numpy(mask)
            self.masks[idx] = mask_tensor

            self._save_masked_images(img_np=np.array(rendered_image_pil),
                                     mask_np=mask,
                                     idx=idx,
                                     color=rand_color)

            # delete to free up memory

            torch.cuda.empty_cache()

            if 'TL_DEBUG' in os.environ:
                break

        # self.initial_rendered_images = self.datamanager.image_batch["image"].clone()
        pass


    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]

        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames or (
                      self.cfg.per_editing_step > 0
                      and self.cfg.edit_begin_step
                      < self.global_step
                      < self.cfg.edit_until_step
                      and self.global_step % self.cfg.per_editing_step == 0
                ):
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )

                    img_edited = result["edit_images"].detach().clone()

                    if cur_index not in self.failed_seg_idx_list:
                        mask_tensor = self.masks[cur_index].to(img_edited.device).unsqueeze(-1)
                        target_image = self.origin_frames[cur_index].to(img_edited.device) * mask_tensor + \
                                       img_edited * (1. - mask_tensor)
                    else:
                        target_image = img_edited.clone()

                    # debug
                    if 'TL_DEBUG' in os.environ:
                        Image.fromarray(self.origin_frames[cur_index][0].mul(255).byte().cpu().numpy()).save(
                            "/home/ps/Documents/0_origin_image.jpg")
                        Image.fromarray(mask_tensor.squeeze().mul(255).byte().cpu().numpy()).save(
                                "/home/ps/Documents/1_mask_image.png")
                        Image.fromarray(img_edited[0].mul(255).byte().cpu().numpy()).save(
                            "/home/ps/Documents/2_edited_image.png")
                        Image.fromarray(target_image[0].mul(255).byte().cpu().numpy()).save(
                            "/home/ps/Documents/3_target_image.jpg")

                    target_image = target_image.to(img_edited.dtype)
                    self.edit_frames[cur_index] = target_image
                    # print("edited image index", cur_index)

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
              self.cfg.loss.lambda_anchor_color > 0
              or self.cfg.loss.lambda_anchor_geo > 0
              or self.cfg.loss.lambda_anchor_scale > 0
              or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
