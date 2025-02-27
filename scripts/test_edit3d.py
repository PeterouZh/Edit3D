import unittest
import os
import sys
from datetime import datetime

from pySciTools.system_run import system_run, get_func_name_and_outdir


class Testing_edit3dgs(unittest.TestCase):

  def test_edit3dgs_face(self, debug=True):
    """
    Example usage of system_run in a test case.

    Setup environment variables and run a sample bash command.
    This example runs `echo $PATH` inside a bash shell.

    Usage:
    export CUDA_VISIBLE_DEVICES=0
    python -c "from scripts.test_edit3d import Testing_edit3dgs;\
      Testing_edit3dgs().test_edit3dgs_face(debug=False)"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # Example: Get the function name and output directory
    func_name, outdir = get_func_name_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)

    if debug:
      os.environ['SCI_DEBUG'] = '1'

    os.environ['PYTHONPATH'] = '.:edit_3dgs/GaussianEditor'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

    cmd_str = f"""
        python edit_3dgs/launch.py
          --config edit_3dgs/configs/edit-n2n-adss.yaml
          --train 
          --gpu 0 
          trainer.max_steps=1500 
          data.source=CACHE_DIR/face
          system.gs_source=CACHE_DIR/train-20231221_105610_663/point_cloud/iteration_7000/point_cloud.ply
          system.prompt_processor.prompt="Turn him into a clown"  
          system.max_densify_percent=0.01 
          system.anchor_weight_init_g0=0.05 
          system.anchor_weight_init=0.1 
          system.anchor_weight_multiplier=1.3 
          system.gs_lr_scaler=5 
          system.gs_final_lr_scaler=5 
          system.color_lr_scaler=5 
          system.opacity_lr_scaler=2 
          system.scaling_lr_scaler=2 
          system.rotation_lr_scaler=2 
          system.loss.lambda_anchor_color=0 
          system.loss.lambda_anchor_geo=50 
          system.loss.lambda_anchor_scale=50 
          system.loss.lambda_anchor_opacity=50 
          system.densify_from_iter=100 
          system.densify_until_iter=1501 
          system.densification_interval=100           
          system.loggers.wandb.enable=false 
          system.loggers.wandb.name="edit_n2n_face_lady_100_den_anchor"          
          system.adss.saved_dir={outdir}   
          system.adss.failed_seg_idx_list=""
          system.adss.ts_caption="A man standing in front of a concrete wall. face. hair. sweater" 
          system.adss.ts_query="face"
          system.adss.ts_use_default_sam=True        
          system.adss.erode_mask=False
          system.adss.erode_kernel_size=10
          system.adss.dilate_mask=True
          system.adss.dilate_kernel_size=10     
          system.seg_prompt="enable"      

        """

    # Run the command using the system_run utility
    start_time = datetime.now()
    system_run(cmd_str)
    end_time = datetime.now()

    elapsed_time = (end_time - start_time).total_seconds()
    print(f"The function execution time is {elapsed_time} seconds")
    print(f"The function execution time is {elapsed_time / 60.} minutes")

  def test_edit3dgs_face_hair(self, debug=True):
    """
    Example usage of system_run in a test case.

    Setup environment variables and run a sample bash command.
    This example runs `echo $PATH` inside a bash shell.

    Usage:
    export CUDA_VISIBLE_DEVICES=0
    python -c "from scripts.test_edit3d import Testing_edit3dgs;\
      Testing_edit3dgs().test_edit3dgs_face_hair(debug=False)"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # Example: Get the function name and output directory
    func_name, outdir = get_func_name_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)

    if debug:
      os.environ['SCI_DEBUG'] = '1'

    os.environ['PYTHONPATH'] = '.:edit_3dgs/GaussianEditor'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

    cmd_str = f"""
        python edit_3dgs/launch.py
          --config edit_3dgs/configs/edit-n2n-adss.yaml
          --train 
          --gpu 0 
          trainer.max_steps=1500 
          data.source=CACHE_DIR/face
          system.gs_source=CACHE_DIR/train-20231221_105610_663/point_cloud/iteration_7000/point_cloud.ply
          system.prompt_processor.prompt="Turn him into a clown"  
          system.max_densify_percent=0.01 
          system.anchor_weight_init_g0=0.05 
          system.anchor_weight_init=0.1 
          system.anchor_weight_multiplier=1.3 
          system.gs_lr_scaler=5 
          system.gs_final_lr_scaler=5 
          system.color_lr_scaler=5 
          system.opacity_lr_scaler=2 
          system.scaling_lr_scaler=2 
          system.rotation_lr_scaler=2 
          system.loss.lambda_anchor_color=0 
          system.loss.lambda_anchor_geo=50 
          system.loss.lambda_anchor_scale=50 
          system.loss.lambda_anchor_opacity=50 
          system.densify_from_iter=100 
          system.densify_until_iter=1501 
          system.densification_interval=100           
          system.loggers.wandb.enable=false 
          system.loggers.wandb.name="edit_n2n_face_lady_100_den_anchor"          
          system.adss.saved_dir={outdir}   
          system.adss.failed_seg_idx_list=""
          system.adss.ts_caption="A man standing in front of a concrete wall. face. hair. sweater" 
          system.adss.ts_query="face . hair"
          system.adss.ts_use_default_sam=True        
          system.adss.erode_mask=False
          system.adss.erode_kernel_size=10
          system.adss.dilate_mask=True
          system.adss.dilate_kernel_size=10     
          system.seg_prompt="enable"          
    
        """

    # Run the command using the system_run utility
    start_time = datetime.now()
    system_run(cmd_str)
    end_time = datetime.now()

    elapsed_time = (end_time - start_time).total_seconds()
    print(f"The function execution time is {elapsed_time} seconds")
    print(f"The function execution time is {elapsed_time / 60.} minutes")
