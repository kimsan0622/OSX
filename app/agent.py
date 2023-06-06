import os
import glob
import logging
import json
import time
import shutil
import subprocess
import functools
from multiprocessing import Pool

from tqdm import tqdm
# from torch.multiprocessing import Process

from model import YoutubeDBManager
from osx_pose_extractor import OSXPoseExtractor

logger = logging.getLogger(__name__)

def cvt_video2images(video_path, output_dir, framerate=30):
    subprocess.run(f"ffmpeg -y -loglevel panic -i {video_path} -r {framerate} {output_dir}/%06d.png", shell=True)

def cvt_video2images_keyframe(video_path, output_dir):
    subprocess.run(f"ffmpeg -y -loglevel panic -vsync 0 -i {video_path} {output_dir}/%06d.png", shell=True)

def cvt_images2video(input_dir, output_path, framerate=30):
    subprocess.run(f"ffmpeg -y -loglevel panic -framerate {framerate} -i {input_dir}/%06d.png -c:v libx264 -pix_fmt yuv420p {output_path}", shell=True)

def muxing_video_and_audio(video_path, audio_path, output_path):
    subprocess.run(f"ffmpeg -y -loglevel panic -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path}", shell=True)

def get_width_and_height(filename):
    result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True).decode()
    fields = json.loads(result)['streams'][0]
    return fields['width'], fields['height']

def get_fps(filename):
    import subprocess, json

    result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True).decode()
    fields = json.loads(result)['streams'][0]
    fps      = eval(fields['r_frame_rate'])
    return fps

def merge_video(left_video, right_video, output_path):
    lw, lh = get_width_and_height(left_video)
    subprocess.run(f'ffmpeg -y -loglevel panic -i {left_video} -i {right_video} -filter_complex "[0:v]pad=width={lw*2}:height={lh}:x=0:y=0[p];[p][1:v]overlay=shortest=1:x={lw}:y=0[full]" -map [full] {output_path}', shell=True)


class OSXProcAgent():
    def __init__(self):
        super().__init__()
        
        self.pose_dir = os.environ.get("POSE_DIR", "/data/pose")
        self.frame_dir = os.environ.get("FRAME_DIR", "/data/frame")
        self.framerate = int(os.environ.get("FRAME_RATE", "30"))
        self.skip_drawing = bool(os.environ.get("SKIP_DRAWING", "false")=="true")
        self.OSX_PROC_SEEK_DURATION = int(os.environ.get(
            'OSX_PROC_SEEK_DURATION', 
            2
        ))
        
        self.osx_extractor = OSXPoseExtractor()
        self.ydbm = YoutubeDBManager()
        

    def extract_pose(self, clip_id, clip_video_path, clip_audio_path):
        base_dir = clip_video_path.split("/")[-2]
        # base_name, _ = os.path.splitext(os.path.basename(clip_video_path))
        
        pose_output_dir = os.path.join(self.pose_dir, base_dir)
        os.makedirs(pose_output_dir, exist_ok=True)
        pose_path = os.path.join(pose_output_dir, f"{clip_id}-pose.json")
        pose_video_path = os.path.join(pose_output_dir, f"{clip_id}-osx.mp4")
        merged_pose_video_path = os.path.join(pose_output_dir, f"{clip_id}-osx-merged.mp4")
        
        frame_output_base_dir = os.path.join(self.frame_dir, base_dir, clip_id)
        frame_output_dir = os.path.join(frame_output_base_dir, "src_frame")
        frame_output_dir_proc = os.path.join(frame_output_base_dir, "tgt_frame")
        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(frame_output_dir_proc, exist_ok=True)
        
        cvt_video2images(clip_video_path, frame_output_dir, self.framerate)
        src_image_list = list(sorted(list(glob.glob(f"{frame_output_dir}/*.png"))))
        
        clip_pose = []
        
        for image_path in tqdm(src_image_list):
            base_name = os.path.basename(image_path)
            person_mesh_list, person_pose_list, vis_img = self.osx_extractor.extract_pose_from_image(
                image_path,
                [
                    "smplx_mesh_cam", 
                    "smplx_root_pose", 
                    "smplx_body_pose", 
                    "smplx_lhand_pose", 
                    "smplx_rhand_pose", 
                    "smplx_jaw_pose",
                    "smplx_shape", 
                    "smplx_expr", 
                    "cam_trans", 
                ]
            )
            self.osx_extractor.save_image(os.path.join(frame_output_dir_proc, base_name), vis_img)
            clip_pose.append(person_pose_list)
        json.dump(clip_pose, open(pose_path, "w"))
        
        # create video for preview
        tmp_pose_video_path = os.path.join(frame_output_dir_proc, f"{clip_id}.mp4")
        cvt_images2video(frame_output_dir_proc, tmp_pose_video_path, self.framerate)
        muxing_video_and_audio(
            tmp_pose_video_path, 
            clip_audio_path, pose_video_path)
        
        tmp_merged_pose_video_path = os.path.join(frame_output_dir_proc, f"{clip_id}-merged.mp4")
        merge_video(
            clip_video_path,
            tmp_pose_video_path,
            tmp_merged_pose_video_path,
            )
        muxing_video_and_audio(
            tmp_merged_pose_video_path, 
            clip_audio_path, merged_pose_video_path)
        
        # delete everything
        shutil.rmtree(frame_output_base_dir)
        
        return pose_path, pose_video_path, merged_pose_video_path
    
    def extract_pose_batched(self, clip_id, clip_video_path, clip_audio_path):
        base_dir = clip_video_path.split("/")[-2]
        # base_name, _ = os.path.splitext(os.path.basename(clip_video_path))
        
        pose_output_dir = os.path.join(self.pose_dir, base_dir)
        os.makedirs(pose_output_dir, exist_ok=True)
        pose_path = os.path.join(pose_output_dir, f"{clip_id}-pose.json")
        pose_video_path = os.path.join(pose_output_dir, f"{clip_id}-osx.mp4")
        merged_pose_video_path = os.path.join(pose_output_dir, f"{clip_id}-osx-merged.mp4")
        
        frame_output_base_dir = os.path.join(self.frame_dir, base_dir, clip_id)
        frame_output_dir = os.path.join(frame_output_base_dir, "src_frame")
        frame_output_dir_proc = os.path.join(frame_output_base_dir, "tgt_frame")
        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(frame_output_dir_proc, exist_ok=True)
        
        cvt_video2images(clip_video_path, frame_output_dir, self.framerate)
        src_image_list = list(sorted(list(glob.glob(f"{frame_output_dir}/*.png"))))
        
        per_image_data = self.osx_extractor.extract_pose_from_images(
                src_image_list,
                [
                    "smplx_mesh_cam", 
                    "smplx_root_pose", 
                    "smplx_body_pose", 
                    "smplx_lhand_pose", 
                    "smplx_rhand_pose", 
                    "smplx_jaw_pose",
                    "smplx_shape", 
                    "smplx_expr", 
                    "cam_trans", 
                ],
                batch_size=8
            )
        
        if not self.skip_drawing:
            draw_func = functools.partial(OSXPoseExtractor.draw_single_sample, root_dir=frame_output_dir_proc)
            with Pool(processes=8) as pool:
                for i in tqdm(pool.imap_unordered(draw_func, per_image_data), desc="proc..."):
                    pass

            # create video for preview
            tmp_pose_video_path = os.path.join(frame_output_dir_proc, f"{clip_id}.mp4")
            cvt_images2video(frame_output_dir_proc, tmp_pose_video_path, self.framerate)
            muxing_video_and_audio(
                tmp_pose_video_path, 
                clip_audio_path, pose_video_path)
            
            tmp_merged_pose_video_path = os.path.join(frame_output_dir_proc, f"{clip_id}-merged.mp4")
            merge_video(
                clip_video_path,
                tmp_pose_video_path,
                tmp_merged_pose_video_path,
                )
            muxing_video_and_audio(
                tmp_merged_pose_video_path, 
                clip_audio_path, merged_pose_video_path)
        
        clip_pose = [res_list for _, _, res_list, _ in per_image_data]
        for idx in range(len(clip_pose)):
            for per_idx in range(len(clip_pose[idx])):
                del clip_pose[idx][per_idx]["smplx_mesh_cam"]
        json.dump(clip_pose, open(pose_path, "w"))
        
        # delete frames
        shutil.rmtree(frame_output_base_dir)
        
        return pose_path, pose_video_path, merged_pose_video_path
    
        
    
    def do(self):
        youtube_clip = self.ydbm.youtube_clips.get_random_clip_by_undefined_field("osx_state")
        if youtube_clip is not None:
            # marking
            try:
                self.ydbm.youtube_clips.update_undefined_field_by_clip_id(youtube_clip["clip_id"], "osx_state", "proc")
                pose_path, pose_video_path, merged_pose_video_path = self.extract_pose_batched(
                    youtube_clip["clip_id"],
                    youtube_clip["clip_video_path"],
                    youtube_clip["clip_audio_path"],
                )
                self.ydbm.youtube_clips.update_undefined_field_by_clip_id(
                        youtube_clip["clip_id"], 
                        "osx_state", "done",
                        pose_path=pose_path,
                        pose_video_path=pose_video_path,
                        merged_pose_video_path=merged_pose_video_path,
                    )
            except Exception as e:
                self.ydbm.youtube_clips.update_undefined_field_by_clip_id(
                        youtube_clip["clip_id"], 
                        "osx_state", None,
                        osx_error=str(e),
                    )
    
    def run(self):
        while True:
            try:
                self.do()
            except Exception as e:
                logger.error(e)
                
            time.sleep(self.OSX_PROC_SEEK_DURATION)
    

if __name__=="__main__":
    osx_proc_agent = OSXProcAgent()

    start = time.time()
    # osx_proc_agent.extract_pose(
    #     "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-0",
    #     "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0.mp4",
    #     "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0.mp3",
    # )
    
    osx_proc_agent.extract_pose_batched(
        "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-15",
        "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-15.mp4",
        "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-15.mp3",
    )
    print(time.time() - start)

    # ydbm = YoutubeDBManager()
    # doc = ydbm.youtube_clips.get_random_clip_by_state_and_undefined_field("new", "osx_state")
    # print(doc)
    # clip_id = doc["clip_id"]
    # clip_video_path = doc["clip_video_path"]
    # clip_audio_path = doc["clip_audio_path"]

    # POSE_DIR = "/data/pose"
    # FRAME_DIR = "/data/frame"
    # FRAME_RATE = 30

    # clip_id = "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-0"
    # clip_video_path = "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0.mp4"
    # clip_audio_path = "/data/clip/687/68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0.mp3"

    # base_dir = clip_video_path.split("/")[-2]
    # base_name, _ = os.path.splitext(os.path.basename(clip_video_path))

    # frame_output_dir = os.path.join(FRAME_DIR, "origin", clip_id)
    # frame_output_dir_proc = os.path.join(FRAME_DIR, "proc", clip_id)
    # os.makedirs(frame_output_dir, exist_ok=True)
    # os.makedirs(frame_output_dir_proc, exist_ok=True)
    # # cvt_video2images(clip_video_path, frame_output_dir, FRAME_RATE)
    # cvt_video2images(clip_video_path, frame_output_dir)
    # # cvt_video2images_keyframe(clip_video_path, frame_output_dir)
    # fps = get_fps(clip_video_path)

    # cvt_images2video(frame_output_dir, "./test.mp4")
    # # cvt_images2video(frame_output_dir, "./test.mp4", fps)
    # merge_video(
    #     clip_video_path,
    #     "./test.mp4",
    #     "test2.mp4",
    #     )
    # muxing_video_and_audio(
    #     "test2.mp4", 
    #     clip_audio_path, "./test3.mp4")


    # src_image_list = list(sorted(list(glob.glob(f"{frame_output_dir}/*.png"))))
    # osx_pose_extractor.extract_pose_from_image(src_image_list[0])

    # for image_path in src_image_list:
    #     base_name = os.path.basename(image_path)
    #     person_mesh_list, person_inf_list, vis_img = osx_pose_extractor.extract_pose_from_image(image_path)
    #     osx_pose_extractor.save_image(os.path.join(frame_output_dir_proc, base_name), vis_img)


    # cvt_images2video(frame_output_dir_proc, "./test.mp4")
    # muxing_video_and_audio("./test.mp4", clip_audio_path, "./test2.mp4")

    # DATA_ROOT="/data"
    # video_path_list = glob.glob(f"{DATA_ROOT}/clip/*/*.mp4")

    # merge_video(
    #     "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0.mp4",
    #     "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0-proc.mp4",
    #     "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0-merged.mp4",
    #     )
    # muxing_video_and_audio(
    #     "68747470733a2f2f796f7574752e62652f4d5456724e633043733130-clip-0-merged.mp4", 
    #     clip_audio_path, "./test3.mp4")