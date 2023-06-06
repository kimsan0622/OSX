import os
import sys
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
sys.path.insert(0, osp.join(os.environ.get("OSX_ROOT_PATH", ".."), 'main'))
sys.path.insert(0, osp.join(os.environ.get("OSX_ROOT_PATH", ".."), 'data'))
from config import cfg
import cv2

GPU_ID = os.environ.get("GPU_ID", "0")
ENCODER_SETTING = os.environ.get("ENCODER_SETTING", "osx_l")
DECODER_SETTING = os.environ.get("DECODER_SETTING", "normal")
PRETRAINED_MODEL_PATH = os.environ.get("PRETRAINED_MODEL_PATH", "../pretrained_models/osx_l.pth.tar")

cfg.set_args(GPU_ID)
cfg.set_additional_args(
    encoder_setting=ENCODER_SETTING, 
    decoder_setting=DECODER_SETTING, 
    pretrained_model_path=PRETRAINED_MODEL_PATH
)

from common.base import Demoer
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x


class OSXPoseExtractor():
    def __init__(self) -> None:
        self.demoer = Demoer()
        self.demoer._make_model()
        self.demoer.model.eval()
        
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        self.transform = transforms.ToTensor()
    
    @torch.no_grad()
    def extract_bboxes(self, image):
        return self.detector(image)
    
    def get_person_result(self, image):
        results = self.extract_bboxes(image)
        person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        class_ids, confidences, boxes = [], [], []
        for detection in person_results:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return boxes, indices
    
    def get_obj_str(self, v, f):
        obj_str = ""
        for i in range(len(v)):
            obj_str += 'v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n'
        for i in range(len(f)):
            obj_str +=  'f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n'
        return obj_str
    
    def cvt_tensor2ndarray(self, data):
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    
    def cvt_ndarray2array(self, data):
        return {k: v.tolist() for k, v in data.items()}
    
    def cvt_tensor2array(self, data):
        return self.cvt_ndarray2array(self.cvt_tensor2ndarray(data))
    
    def create_bboxes(self, image_path_list):
        image_seq_list = []
        image_list = []
        for img_seq, image_path in enumerate(image_path_list):
            original_img = load_img(image_path)
            original_img_height, original_img_width = original_img.shape[:2]
            boxes, indices = self.get_person_result(original_img)
            for num, indice in enumerate(indices):
                bbox = boxes[indice]  # x,y,h,w
                bbox = process_bbox(bbox, original_img_width, original_img_height)
                img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                image_seq_list.append((img_seq, bbox, image_path))
                image_list.append(img)
        return image_seq_list, image_list
    
    def batchfy_data(self, data_list, batch_size=8):
        return [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]
    
    def unbatchfy_result(self, batched_dict):
        batched_dict_key = list(batched_dict.keys())
        v_len = len(batched_dict[batched_dict_key[0]])

        list_of_dict = []
        for idx in range(v_len):
            list_of_dict.append({k:batched_dict[k][idx] for k in batched_dict_key})
        return list_of_dict
    
    def extract_pose_from_images(self, image_path_list, keep_data=None, batch_size=8):
        image_seq_list, image_list = self.create_bboxes(image_path_list)
        image_list_batched = self.batchfy_data(image_list, batch_size)
        
        batched_res = []
        for batched_imgs in tqdm(image_list_batched, "batched_proc..."):
            imgs = [self.transform(img.astype(np.float32))/255 for img in batched_imgs]
            imgs = torch.stack(imgs).cuda()
            inputs = {'img': imgs}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = self.demoer.model(inputs, targets, meta_info, 'test')
            
            fmt_out = out
            if keep_data is not None:
                fmt_out = {k:v for k, v in fmt_out.items() if k in keep_data}
            det_res = self.cvt_tensor2array(fmt_out)
            batched_res.extend(self.unbatchfy_result(det_res))
        
        batched_tuples = []
        per_image_data = []
        for (img_seq, bbox, img_path), res in zip(image_seq_list, batched_res):
            if len(per_image_data) == 0:
                per_image_data = [img_path, [bbox], [res], img_seq]
            elif per_image_data[-1] == img_seq:
                per_image_data[-2].append(res)
                per_image_data[1].append(bbox)
            else:
                batched_tuples.append(per_image_data)
                per_image_data = [img_path, [bbox], [res], img_seq]
        if len(per_image_data) > 0:
            batched_tuples.append(per_image_data)
        
        batched_tuples_dict = {v[0]:v for v in batched_tuples}
        batched_tuples = []
        for img_seq, img_path in enumerate(image_path_list):
            if img_path in batched_tuples_dict:
                batched_tuples.append(batched_tuples_dict[img_path])
            else:
                batched_tuples.append([img_path, [], [], img_seq])
        return batched_tuples
    
    def draw_samples(self, per_image_data, root_dir):
        for img_path, bbox_list, res_list, _ in tqdm(per_image_data, desc="draw images..."):
            base_name = os.path.basename(img_path)
            vis_img = load_img(img_path)
            for bbox, res in zip(bbox_list, res_list):
                mesh = np.array(res['smplx_mesh_cam'])
                focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
                vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
            cv2.imwrite(os.path.join(root_dir, base_name), vis_img[:, :, ::-1])
    
    @staticmethod
    def draw_single_sample(data_tuple, root_dir=""):
        cfg_focal = (5000, 5000)  # virtual focal lengths
        cfg_input_body_shape = (256, 192)
        cfg_princpt = (cfg_input_body_shape[1] / 2, cfg_input_body_shape[0] / 2)  # virtual principal point position
        
        img_path, bbox_list, res_list, _  = data_tuple
        
        base_name = os.path.basename(img_path)
        vis_img = load_img(img_path)
        for bbox, res in zip(bbox_list, res_list):
            mesh = np.array(res['smplx_mesh_cam'])
            focal = [cfg_focal[0] / cfg_input_body_shape[1] * bbox[2], cfg_focal[1] / cfg_input_body_shape[0] * bbox[3]]
            princpt = [cfg_princpt[0] / cfg_input_body_shape[1] * bbox[2] + bbox[0], cfg_princpt[1] / cfg_input_body_shape[0] * bbox[3] + bbox[1]]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        cv2.imwrite(os.path.join(root_dir, base_name), vis_img[:, :, ::-1])
    

    def extract_pose_from_image(self, image_path, keep_data=None):
        original_img = load_img(image_path)
        original_img_height, original_img_width = original_img.shape[:2]
        
        boxes, indices = self.get_person_result(original_img)
        vis_img = original_img.copy()
        
        person_mesh_list = []
        person_inf_list = []
        
        for num, indice in enumerate(indices):
            bbox = boxes[indice]  # x,y,h,w
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = self.transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = self.demoer.model(inputs, targets, meta_info, 'test')
            
            fmt_out = out
            if keep_data is not None:
                fmt_out = {k:v for k, v in fmt_out.items() if k in keep_data}
            det_res = self.cvt_tensor2array(fmt_out)
            
            person_inf_list.append(det_res)

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
            mesh = mesh[0]
            

            # save mesh
            # person_mesh_list.append(self.get_obj_str(mesh, smpl_x.face))

            # render mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        return person_mesh_list, person_inf_list, vis_img[:, :, ::-1]
    
    def save_image(self, image_path, image):
        cv2.imwrite(image_path, image)