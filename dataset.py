import sys

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_mesh_tensor
from audiodvp_utils import util
from torch.utils.data import Dataset
from natsort import natsorted
import pickle
from tqdm import tqdm

class Lipsync3DMeshTestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = os.path.join(opt.tgt_dir, opt.tgt_folder_name)

        self.stabilized_mesh = [os.path.join(self.tgt_dir, 'stabilized_norm_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'stabilized_norm_mesh')))]

        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')

        self.reference_mesh = landmarkdict_to_mesh_tensor(torch.load(os.path.join(opt.tgt_dir, 'reference_mesh.pt')))

        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        
        self.mesh_dict_list = util.load_coef(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.image_filenames = util.get_file_list(os.path.join(self.tgt_dir, 'crop'))
        
        with open(os.path.join(self.tgt_dir, 'face_emb.pkl'), 'rb') as f:
            self.face_embedding = pickle.load(f)

        print('Inference set size: ', len(self.filenames))

    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def __getitem__(self, index):
        if index == (min(self.audio_stft.shape[2] // 4, len(self.filenames))-1):
            nex_index = index
        else:
            nex_index = index + 1

        # audio selection

        audio_cur_index = index * 4
        audio_nex_index = nex_index * 4

        def returnFeature(index):
            audio_feature_list = []
            for i in range(index - 12, index + 12):
                if i < 0:
                    audio_feature_list.append(self.audio_stft[:, :, 0])
                elif i >= self.audio_stft.shape[2]:
                    audio_feature_list.append(self.audio_stft[:, :, -1])
                else:
                    audio_feature_list.append(self.audio_stft[:, :, i])\

            return audio_feature_list

        audio_cur_feature_list = returnFeature(audio_cur_index)
        audio_nex_feature_list = returnFeature(audio_nex_index)

        def returnTorchAudio(audio_feature_list):
            return torch.stack(audio_feature_list, 2)

        audio_cur_feature = returnTorchAudio(audio_cur_feature_list)
        audio_nex_feature = returnTorchAudio(audio_nex_feature_list)

        # Face Embedding

        face_emb_img_name = self.image_filenames[0]

        cur_face_embedding = self.face_embedding[os.path.basename(face_emb_img_name)]
        nex_face_embedding = self.face_embedding[os.path.basename(face_emb_img_name)]
  
        # GT Mesh
        cur_gt_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))
        cur_gt_mesh = cur_gt_mesh - self.reference_mesh
        nex_gt_mesh = torch.tensor(torch.load(self.stabilized_mesh[nex_index]))
        nex_gt_mesh = nex_gt_mesh - self.reference_mesh

        # Emotional State

        def emotionRange(num):
            emotion = torch.zeros(7)
            if 1 <= num <= 50:
                emotion[0] = 1.
            elif 51 <= num <= 100:
                emotion[1] = 1.
            elif 101 <= num <= 150:
                emotion[2] = 1.
            elif 151 <= num <= 200:
                emotion[3] = 1.
            elif 201 <= num <= 250:
                emotion[4] = 1.
            elif 251 <= num <= 300:
                emotion[5] = 1.
            elif 301 <= num <= 350:
                emotion[6] = 1.
            return emotion

        video_name = self.tgt_dir.split('/')[-1]
        emotion_num = int(video_name.split('-')[1])
        cur_emotion = emotionRange(emotion_num)
        nex_emotion = emotionRange(emotion_num)

        filename = os.path.basename(self.filenames[index])

        return {
            'audio_feature': audio_cur_feature, 'gt_mesh': cur_gt_mesh,
            'face_emb' : cur_face_embedding, 'emotion': cur_emotion,
            'audio_feature_nxt': audio_nex_feature, 'gt_mesh_nxt': nex_gt_mesh,
            'face_emb_nxt' : nex_face_embedding, 'emotion_nxt': nex_emotion,
            'reference_mesh' : self.reference_mesh, 'filename':filename
        }



class Lipsync3DMeshDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_dir = opt.tgt_dir

        # TODO 현재, 다음 index audio feature
                    #    gt mesh
                    #    face embedding - dictionary format으로 key 이미지 image
                    #    emotion state
        videos = [x for x in natsorted(os.listdir(self.data_dir)) if os.path.isdir(os.path.join(self.data_dir, x)) and 'checkpoint' not in x and 'test' not in x]
        self.file_index = 0
        self.batch_counter = 0
        self.reference_mesh = landmarkdict_to_mesh_tensor(torch.load(os.path.join(opt.tgt_dir, 'reference_mesh.pt')))

        for video in tqdm(videos):
            
            stft_path = os.path.join(self.data_dir, video, 'audio/audio_stft.pt')

            if not os.path.exists(stft_path):
                audio = librosa.load(os.path.join(self.data_dir, video, 'audio/audio.wav'),16000)[0]
                audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
                audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
                torch.save(audio_stft, os.path.join(self.data_dir, video, 'audio/audio_stft.pt'))
        
        train_idx = int(len(videos) * opt.train_rate)
        
        if opt.isTrain:
            self.dataset = videos[:train_idx]
        else:
            self.dataset = videos[train_idx,:]

        self.max_file_index = len(self.dataset)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
    
    # Video selection
        
        dataset = self.dataset

        video = dataset[self.file_index]

        with open(os.path.join(self.opt.tgt_dir, video, 'face_emb.pkl'), 'rb') as f:
            face_embedding = pickle.load(f)

        audio = torch.load(os.path.join(self.data_dir, video, 'audio/audio_stft.pt'))
        stabilized_mesh = [os.path.join(self.data_dir, video, 'stabilized_norm_mesh', x) for x in natsorted(os.listdir(os.path.join(self.data_dir, video, 'stabilized_norm_mesh')))]
        filenames = util.get_file_list(os.path.join(self.data_dir, video, 'stabilized_norm_mesh'))
        image_filenames = util.get_file_list(os.path.join(self.data_dir, video, 'crop'))

    # current and next index selection of a video
        max_index = len(filenames)
        cur_index = self.batch_counter

        if cur_index == max_index - 1:
            nex_index = cur_index
            self.batch_counter = 0
            self.file_index = (self.file_index + 1) % self.max_file_index
        else:
            nex_index = cur_index + 1
            self.batch_counter += 1

        
    # Audio feature extraction
        audio_cur_idx = cur_index * 4
        audio_nex_idx = nex_index * 4

        def returnFeature(index):
            audio_feature_list = []
            for i in range(index - 12, index + 12):
                if i < 0:
                    audio_feature_list.append(audio[:, :, 0])
                elif i >= audio.shape[2]:
                    audio_feature_list.append(audio[:, :, -1])
                else:
                    audio_feature_list.append(audio[:, :, i])

            return audio_feature_list
        audio_cur_feature_list = returnFeature(audio_cur_idx)
        audio_nex_feature_list = returnFeature(audio_nex_idx)

        def returnTorchAudio(audio_feature_list):
            return torch.stack(audio_feature_list, 2)

        audio_cur_feature = returnTorchAudio(audio_cur_feature_list)
        audio_nex_feature = returnTorchAudio(audio_nex_feature_list)

    # Face Embedding
        face_emb_img_name = image_filenames[cur_index]
        next_face_emb = image_filenames[nex_index]

        cur_face_embedding = face_embedding[os.path.basename(face_emb_img_name)]
        nex_face_embedding = face_embedding[os.path.basename(next_face_emb)]
    
    # Ground Truth Mesh

        cur_gt_mesh = torch.tensor(torch.load(stabilized_mesh[cur_index]))
        cur_gt_mesh = cur_gt_mesh - self.reference_mesh
        nex_gt_mesh = torch.tensor(torch.load(stabilized_mesh[nex_index]))
        nex_gt_mesh = nex_gt_mesh - self.reference_mesh

    # Emotional State

        def emotionRange(num):
            emotion = torch.zeros(7)
            if 1 <= num <= 50:
                emotion[0] = 1.
            elif 51 <= num <= 100:
                emotion[1] = 1.
            elif 101 <= num <= 150:
                emotion[2] = 1.
            elif 151 <= num <= 200:
                emotion[3] = 1.
            elif 201 <= num <= 250:
                emotion[4] = 1.
            elif 251 <= num <= 300:
                emotion[5] = 1.
            elif 301 <= num <= 350:
                emotion[6] = 1.
            return emotion

        emotion_num = int(video.split('-')[1])
        cur_emotion = emotionRange(emotion_num)
        nex_emotion = emotionRange(emotion_num)

        return {
            'audio_feature': audio_cur_feature, 'gt_mesh': cur_gt_mesh,
            'face_emb' : cur_face_embedding, 'emotion': cur_emotion,
            'audio_feature_nxt': audio_nex_feature, 'gt_mesh_nxt': nex_gt_mesh,
            'face_emb_nxt' : nex_face_embedding, 'emotion_nxt': nex_emotion
        }