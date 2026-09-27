"""Image-mode MediaPipe person/face/hand inference; missing detections stay missing."""
import argparse
from pathlib import Path
import urllib.request
import numpy as np
from run_depth_job import read_job,package

MODELS={
 'pose':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
 'face':'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
 'hand':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'}


def run(job_path,result_path,models):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    from PIL import Image
    job,image=read_job(job_path,'mediapipe');root=Path(models);root.mkdir(parents=True,exist_ok=True)
    for key,url in MODELS.items():
        path=root/(key+'.task')
        if not path.is_file():
            temp=path.with_suffix('.tmp');urllib.request.urlretrieve(url,temp);temp.replace(path)
    bg=Image.new('RGBA',image.size,'white');bg.alpha_composite(image)
    rgb=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(bg.convert('RGB')).copy())
    arrays={};metadata={'inferred':True,'sourceSize':list(image.size),'warnings':[],'backendVersion':mp.__version__}
    def landmarks(values):
        return [[p.x,p.y,p.z,float(p.visibility or 0),float(p.presence or 0)] for p in values]
    with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'pose.task')),output_segmentation_masks=True,num_poses=1,min_pose_detection_confidence=.2,min_pose_presence_confidence=.2)) as detector:
        result=detector.detect(rgb)
        metadata['pose']=landmarks(result.pose_landmarks[0]) if result.pose_landmarks else []
        if metadata['pose']:
            pose=np.asarray(metadata['pose'])
            if np.linalg.norm(pose[11,:2]-pose[12,:2])<.04 or np.ptp(pose[:25,1])<.12:
                metadata['rejectedPose']=metadata['pose'];metadata['pose']=[]
                metadata['warnings'].append('Collapsed pose geometry rejected. Set shoulder/elbow/wrist joints manually.')
        if result.segmentation_masks:
            mask=result.segmentation_masks[0].numpy_view().copy()
            if mask.ndim==3 and mask.shape[2]==1:mask=mask[:,:,0]
            arrays['person-mask.npy']=mask
            if mask.max()<.5:metadata['warnings'].append('Person mask confidence is below 0.5; do not use it to erase the source without review.')
    with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'face.task')),num_faces=1,output_face_blendshapes=True)) as detector:
        result=detector.detect(rgb);metadata['face']=landmarks(result.face_landmarks[0]) if result.face_landmarks else []
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'hand.task')),num_hands=2,min_hand_detection_confidence=.3,min_hand_presence_confidence=.3)) as detector:
        result=detector.detect(rgb);metadata['hands']=[landmarks(x) for x in result.hand_landmarks]
    for key in ('pose','face','hands'):
        if not metadata[key]:metadata['warnings'].append(key+' was not detected; correct manually. Anime accuracy is not guaranteed.')
    package(job,result_path,metadata,arrays)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('result');p.add_argument('--models',default='models/mediapipe')
    a=p.parse_args();run(a.job,a.result,a.models)
