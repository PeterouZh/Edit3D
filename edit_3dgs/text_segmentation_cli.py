import time
from PIL import Image
import random
import numpy as np
import tyro
import cv2
from scipy.ndimage import filters
from skimage import transform as skimage_transform

from edit_3dgs import tyro_utils

import lavis.processors
import lavis.models
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import segment_anything
# import mobile_sam
import segment_anything as mobile_sam

def resize_longest_side(image,
                        img_size=768):
  height, width = image.shape[:2]

  # 确定最长边
  if height > width:
    new_height = img_size
    new_width = int(width * (img_size / height))
  else:
    new_width = img_size
    new_height = int(height * (img_size / width))

  # 使用cv2.resize来resize图像
  resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
  return resized_image


def getAttMap_gray(img,
                   attMap,
                   blur=True,
                   overlap=True,
                   thresh=200):
  attMap -= attMap.min()
  if attMap.max() > 0:
    attMap /= attMap.max()
  attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
  if blur:
    attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
    attMap -= attMap.min()
    attMap /= attMap.max()

  # Create a mask for the brightest regions
  _, attMap = cv2.threshold(attMap, thresh/255., 1., cv2.THRESH_BINARY)

  if overlap:
    attMap = (attMap).reshape(attMap.shape + (1,)) * img

  return attMap


def build_sam(model_type,
              sam_checkpoint,
              use_default_sam,
              points_per_side,
              pred_iou_thresh,
              stability_score_thresh,
              crop_n_layers,
              crop_n_points_downscale_factor,
              min_mask_region_area,
              device):

  sam = segment_anything.sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)
  if use_default_sam:
    mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)

  else:
    mask_generator = segment_anything.SamAutomaticMaskGenerator(
      model=sam,
      points_per_side=points_per_side,
      pred_iou_thresh=pred_iou_thresh,
      stability_score_thresh=stability_score_thresh,
      crop_n_layers=crop_n_layers,
      crop_n_points_downscale_factor=crop_n_points_downscale_factor,
      min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
    )

  return mask_generator


def build_mobile_sam(model_type,
                     sam_checkpoint,
                     use_default_sam,
                     points_per_side,
                     pred_iou_thresh,
                     stability_score_thresh,
                     crop_n_layers,
                     crop_n_points_downscale_factor,
                     min_mask_region_area,
                     device):

  sam = mobile_sam.sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)
  if use_default_sam:
    mask_generator = mobile_sam.SamAutomaticMaskGenerator(sam)

  else:
    mask_generator = mobile_sam.SamAutomaticMaskGenerator(
      model=sam,
      points_per_side=points_per_side,
      pred_iou_thresh=pred_iou_thresh,
      stability_score_thresh=stability_score_thresh,
      crop_n_layers=crop_n_layers,
      crop_n_points_downscale_factor=crop_n_points_downscale_factor,
      min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
    )

  return mask_generator


def relative_intersection(mask, rela_mask):
  intersection = np.logical_and(mask, rela_mask)
  ri = intersection.sum() / float(rela_mask.sum())
  return ri


def filter_sam_masks(masks,
                     thresh):
  masks_filtered = []
  sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

  for idx_i in range(len(masks)):
    discard = False
    for idx_j in range(idx_i + 1, len(masks)):
      ri = relative_intersection(mask=sorted_masks[idx_i]['segmentation'],
                                 rela_mask=sorted_masks[idx_j]['segmentation'])
      if ri >= thresh:
        discard = True
        break
    if not discard:
      masks_filtered.append(sorted_masks[idx_i])

  return masks_filtered

def merge_sam_masks(masks,
                    min_area=100):
  masks_merged = []

  if len(masks) > 0:
    stacked_masks = np.stack([mask['segmentation'] * mask['area'] for idx, mask in enumerate(masks)])
    mask_acc = stacked_masks.sum(axis=0)

    for value in np.unique(mask_acc):
      cur_mask = {}
      cur_mask['segmentation'] = (mask_acc == value)
      cur_mask['area'] = cur_mask['segmentation'].sum()
      if cur_mask['area'] > min_area:
        masks_merged.append(cur_mask)

  return masks_merged

class TextSegmentation(object):

  def __init__(self,
               img_size,
               caption,
               blip_type,
               layer_num,
               blur,
               thresh,
               model_type,
               sam_checkpoint,
               use_default_sam,
               points_per_side,
               pred_iou_thresh,
               stability_score_thresh,
               crop_n_layers,
               crop_n_points_downscale_factor,
               min_mask_region_area,
               SAM_masks_filter_thresh,
               min_area,
               use_mobilesam,
               mobilesam_model_type,
               mobilesam_ckpt,
               device='cuda',
               **kwargs):
    # self.img_size = img_size
    self.caption = caption
    self.layer_num = layer_num
    self.blur = blur
    self.thresh = thresh
    self.SAM_masks_filter_thresh = SAM_masks_filter_thresh
    self.min_area = min_area

    self.device = device

    self.vis_processor = lavis.processors.load_processor("blip_image_eval").build(image_size=384)
    self.text_processor = lavis.processors.load_processor("blip_caption")
    self.tokenizer = lavis.models.BlipBase.init_tokenizer()

    self.model = lavis.models.load_model(
      name="blip_image_text_matching", model_type=blip_type, is_eval=True, device=device)

    if use_mobilesam:
      self.mask_generator = build_mobile_sam(model_type=mobilesam_model_type,
                                             sam_checkpoint=mobilesam_ckpt,
                                             use_default_sam=use_default_sam,
                                             points_per_side=points_per_side,
                                             pred_iou_thresh=pred_iou_thresh,
                                             stability_score_thresh=stability_score_thresh,
                                             crop_n_layers=crop_n_layers,
                                             crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                             min_mask_region_area=min_mask_region_area,
                                             device=device)
    else:
      self.mask_generator = build_sam(model_type=model_type,
                                      sam_checkpoint=sam_checkpoint,
                                      use_default_sam=use_default_sam,
                                      points_per_side=points_per_side,
                                      pred_iou_thresh=pred_iou_thresh,
                                      stability_score_thresh=stability_score_thresh,
                                      crop_n_layers=crop_n_layers,
                                      crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                      min_mask_region_area=min_mask_region_area,
                                      device=device)

    pass

  def text_seg(self,
               img_pil,
               query,
               img_size=None):
    image_np = np.array(img_pil).copy()
    if img_size is not None:
      image_np = resize_longest_side(image_np, img_size=img_size)
    # norm_img = image_np / 255.

    img_colored_mask = self.vis_processor(img_pil).unsqueeze(0).to(self.device)

    query = f"{self.caption} {query}"
    qry = self.text_processor(query)
    qry_tok = self.tokenizer(qry, return_tensors="pt").to(self.device)

    gradcam, _ = compute_gradcam(self.model,
                                 img_colored_mask,
                                 qry,
                                 qry_tok,
                                 block_num=self.layer_num)

    gradcam_iter = gradcam[0][2:-1]
    token_id_iter = qry_tok.input_ids[0][1:-1]

    gradcam_img = gradcam_iter[-1]
    token_id = token_id_iter[-1]

    # word = self.tokenizer.decode([token_id])
    gradcam_todraw = getAttMap_gray(img=image_np,
                                    attMap=gradcam_img.numpy().astype(np.float32),
                                    blur=self.blur,
                                    overlap=False,
                                    thresh=self.thresh)
    gradcam_mask = (gradcam_todraw > 0)

    masks = self.mask_generator.generate(image_np)
    # masks_filtered = filter_sam_masks(masks, thresh=self.SAM_masks_filter_thresh)
    masks_filtered = merge_sam_masks(masks=masks, min_area=self.min_area)

    GRI_max = 0
    idx_max = -1
    for idx, ann in enumerate(masks_filtered):
      sam_mask = ann['segmentation']
      GRI = relative_intersection(mask=sam_mask, rela_mask=gradcam_mask)

      if GRI >= GRI_max:
        GRI_max = GRI
        idx_max = idx

    if idx_max >= 0:
      mask = masks_filtered[idx_max]['segmentation']
    else:
      mask = np.zeros(shape=(*image_np.shape[:2], ), dtype=bool)

    return mask

  def text_seg_multi_queries(self,
                             img_pil,
                             query,
                             img_size=None
                             ):
    image_np = np.array(img_pil).copy()
    if img_size is not None:
      image_np = resize_longest_side(image_np, img_size=img_size)
    # norm_img = image_np / 255.

    img_colored_mask = self.vis_processor(img_pil).unsqueeze(0).to(self.device)

    query_all = f"{self.caption} {query}"
    qry = self.text_processor(query_all)
    qry_tok = self.tokenizer(qry, return_tensors="pt").to(self.device)

    gradcam, _ = compute_gradcam(self.model,
                                 img_colored_mask,
                                 qry,
                                 qry_tok,
                                 block_num=self.layer_num)

    gradcam_iter = gradcam[0][2:-1]
    # token_id_iter = qry_tok.input_ids[0][1:-1]

    gradcam_imgs = gradcam_iter[-len(query.split('.')):]
    # token_id = token_id_iter[-1]
    # word = self.tokenizer.decode([token_id])

    masks = self.mask_generator.generate(image_np)
    # masks_filtered = filter_sam_masks(masks, thresh=self.SAM_masks_filter_thresh)
    masks_filtered = merge_sam_masks(masks=masks, min_area=self.min_area)

    ret_mask = np.zeros(image_np.shape[0:2], dtype=bool)
    for gradcam_img in gradcam_imgs:
      gradcam_todraw = getAttMap_gray(img=image_np,
                                      attMap=gradcam_img.numpy().astype(np.float32),
                                      blur=self.blur,
                                      overlap=False,
                                      thresh=self.thresh)
      gradcam_mask = (gradcam_todraw > 0)

      GRI_max = 0
      idx_max = -1
      for idx, ann in enumerate(masks_filtered):
        sam_mask = ann['segmentation']
        GRI = relative_intersection(mask=sam_mask, rela_mask=gradcam_mask)

        if GRI >= GRI_max:
          GRI_max = GRI
          idx_max = idx

      if idx_max >= 0:
        mask = masks_filtered[idx_max]['segmentation']
      else:
        mask = np.zeros(shape=(*image_np.shape[:2],), dtype=bool)

      ret_mask = np.logical_or(ret_mask, mask)
    return ret_mask

  def set_seed(self, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    pass


def main(cfg_path: str,
         tl_command: str = None):

  global global_cfg
  global_cfg = tyro_utils.parse_cfg_from_yaml_cli(cfg_path=cfg_path,
                                                  sub_key=tl_command)

  device = 'cuda'
  text_seg = TextSegmentation(**global_cfg, device=device)

  img_pil = Image.open(global_cfg.image_path).convert('RGB')

  start_time = time.perf_counter()
  mask = text_seg.text_seg(img_pil=img_pil, query=global_cfg.query)
  end_time = time.perf_counter()

  elapsed_time = end_time - start_time
  print(f"Time: {elapsed_time:.6f}s")

  text_seg.set_seed()
  img_np = np.array(img_pil) / 255.
  img_np = resize_longest_side(img_np, img_size=global_cfg.img_size)
  img_colored_mask = img_np.copy()
  img_colored_mask[mask] = np.random.random(3)
  alpha = 0.9
  image_anns = np.clip((1 - alpha) * img_np + alpha * img_colored_mask, 0, 1.)

  img_anns = Image.fromarray(np.clip(image_anns * 255, 0, 255).astype(np.uint8))
  img_anns.save(f"{global_cfg.tl_outdir}/output.jpg")

  pass


if __name__ == '__main__':
  tyro.cli(main, return_unknown_args=True)
  pass