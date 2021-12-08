from torch import optim
from torch.optim import optimizer
import torch
from torch.utils.data import DataLoader
from options import Options
from dataset import Lipsync3DMeshDataset
from model import Audio2GeometryModel
from loss import L2Loss, MotionLoss
from audiodvp_utils.visualizer import Visualizer
import time
import os


if __name__ == '__main__':
    opt = Options().parse_args()
    device = opt.device

    print("Loading Dataset")
    dataset = Lipsync3DMeshDataset(opt)
    print("Dataset Loaded")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,  # default not shuffle
        num_workers=opt.num_workers,
        drop_last=True,  # the batch size cannot change during the training so the last uncomplete batch need to be dropped
    )

    visualizer = Visualizer(opt)
    model = Audio2GeometryModel().to(device)

    criterionPosition = L2Loss()
    criterionMotion = MotionLoss()

    if opt.load_model:
        if os.path.exists(os.path.join(opt.tgt_dir, opt.model_name)):
            print(os.path.join(opt.tgt_dir, opt.model_name))
            model.load_state_dict(torch.load(os.path.join(opt.tgt_dir, opt.model_name), map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    os.makedirs(os.path.join(opt.tgt_dir, 'checkpoint_{}'.format(opt.gpu_ids)), exist_ok=True)

    total_iters = 0
    for epoch in range(opt.num_epoch):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            optimizer.zero_grad()

            audio_feature = data['audio_feature'].to(device)
            gt_mesh = data['gt_mesh'].to(device)
            face_emb = data['face_emb'].to(device)
            emotion = data['emotion'].to(device)

            audio_feature_nxt = data['audio_feature_nxt'].to(device)
            gt_mesh_nxt = data['gt_mesh_nxt'].to(device)
            face_emb_nxt = data['face_emb_nxt'].to(device)
            emotion_nxt = data['emotion_nxt'].to(device)

            pred_mesh = model(audio_feature, face_emb, emotion)
            pred_mesh_nxt = model(audio_feature_nxt, face_emb_nxt, emotion_nxt)
            PositionLoss = criterionPosition(pred_mesh, gt_mesh) + criterionPosition(pred_mesh_nxt, gt_mesh_nxt)
            MotionLoss = criterionMotion(pred_mesh, pred_mesh_nxt, gt_mesh, gt_mesh_nxt)

            loss = opt.lambda_pos * PositionLoss + opt.lambda_motion * MotionLoss
            loss.backward()
            optimizer.step()

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = {'PositionLoss': PositionLoss, 'MotionLoss' : MotionLoss}
                visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
                visualizer.plot_current_losses(total_iters, losses)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

        if epoch % opt.checkpoint_interval == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(opt.tgt_dir, 'checkpoint_{}'.format(opt.gpu_ids), 'checkpoint_{}.pth'.format(epoch)))
            print("Checkpoint saved")

    torch.save(model.state_dict(), os.path.join(opt.tgt_dir, 'Audio2GeometryModel_{}.pth'.format(opt.gpu_ids)))