import torch
from options import Options
from dataset import Lipsync3DMeshTestDataset
from model import Audio2GeometryModel
from loss import L2Loss, MotionLoss
import time
from utils import mesh_tensor_to_landmarkdict, draw_mesh_images
import os
from tqdm import tqdm
import cv2
import shutil

if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device
    calculate_test_loss = (opt.src_dir == opt.tgt_dir)
    dataset = Lipsync3DMeshTestDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    model = Audio2GeometryModel().to(device)
    criterionPosition = L2Loss()
    criterionMotion = MotionLoss()

    def emptyFolder(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    ckpt = torch.load(os.path.join(opt.tgt_dir, opt.model_name), map_location=device)
    model.load_state_dict(ckpt)
    
    emptyFolder(os.path.join(opt.src_dir, 'reenact_mesh'))
    emptyFolder(os.path.join(opt.src_dir, 'reenact_mesh_image'))
    
    avg_Positionloss = 0
    avg_Motionloss = 0

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(tqdm(test_dataloader)):
            audio_feature = data['audio_feature'].to(device)
            gt_mesh = data['gt_mesh'].to(device)
            face_emb = data['face_emb'].to(device)
            emotion = data['emotion'].to(device)
            reference_mesh = data['reference_mesh'].to(device)
            
            filename = data['filename'][0]

            pred_mesh = model(audio_feature, face_emb, emotion=emotion)

            if calculate_test_loss and (i > int(len(test_dataloader) * opt.train_rate)):
                audio_feature_nxt = data['audio_feature_nxt'].to(device)
                gt_mesh_nxt = data['gt_mesh_nxt'].to(device)
                face_emb_nxt = data['face_emb_nxt'].to(device)
                emotion_nxt = data['emotion_nxt'].to(device)
                pred_mesh_nxt = model(audio_feature_nxt, face_emb_nxt, emotion_nxt)
                PositionLoss = criterionPosition(pred_mesh, gt_mesh) + criterionPosition(pred_mesh_nxt, gt_mesh_nxt)
                MotionLoss = criterionMotion(pred_mesh, pred_mesh_nxt, gt_mesh, gt_mesh_nxt)
                avg_Positionloss += PositionLoss.detach() / int(len(test_dataloader) * (1 - opt.train_rate))
                avg_Motionloss += MotionLoss.detach() / int(len(test_dataloader) * (1 - opt.train_rate))

            pred_real_mesh = reference_mesh + pred_mesh
            pred_real_mesh = pred_real_mesh[0].cpu().detach()
            landmark_dict = mesh_tensor_to_landmarkdict(pred_real_mesh)

            torch.save(landmark_dict, os.path.join(opt.src_dir, 'reenact_mesh', filename))
    
    if calculate_test_loss:
        print('Average Test Position loss : ', avg_Positionloss)
        print('Average Test Motion loss : ', avg_Motionloss)

    print('Start drawing reenact mesh')
    image = cv2.imread(os.path.join(opt.tgt_dir, 'reference_frame.jpg'))
    image_rows, image_cols, _ = image.shape
    draw_mesh_images(os.path.join(opt.src_dir, 'reenact_mesh'), os.path.join(opt.src_dir, 'reenact_mesh_image'), image_rows, image_cols)
    os.makedirs(os.path.join(opt.src_dir, 'results'), exist_ok=True)
    os.system('ffmpeg -y -i {}/%d.jpg -i {} -c:v libx264 -crf 1 -r 25 {}'.format(os.path.join(opt.src_dir, 'reenact_mesh_image'), os.path.join(opt.src_dir, 'audio','audio.wav'), os.path.join(opt.src_dir, 'results', '{}.mp4'.format(opt.output_vid_name))))
    os.system('ffmpeg -y -i {}/%d.png -i {} -c:v libx264 -crf 1 -r 25 {}'.format(os.path.join(opt.src_dir, 'mesh_norm_image'), os.path.join(opt.src_dir, 'audio', 'audio.wav'), os.path.join(opt.src_dir, 'results', 'original.mp4')))
    os.system('ffmpeg -y -i {} -i {} -filter_complex hstack=inputs=2 {}'.format(os.path.join(opt.src_dir, 'results', '{}.mp4'.format(opt.output_vid_name)), os.path.join(opt.src_dir, 'results', 'original.mp4'), os.path.join(opt.src_dir, 'results', 'comparisons.mp4')))