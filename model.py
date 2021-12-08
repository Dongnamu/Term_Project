from cv2 import getOptimalNewCameraMatrix
import torch
import torch.nn as nn

class Audio2GeometryModel(nn.Module):
    def __init__(self):
        super(Audio2GeometryModel, self).__init__()

        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=72, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),   # 2 x 256 x 24 -> 72 x 128 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=72,out_channels=108, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 72 x 128 x 24 -> 108 x 64 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=108,out_channels=162, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 108 x 64 x 24 -> 162 x 32 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=162,out_channels=243, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dilation=1),    # 162 x 32 x 24 -> 243 x 16 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=243,out_channels=256, kernel_size=(4, 1), stride=(4, 1), padding=(1, 0), dilation=1),    # 243 x 16 x 24 -> 256 x 4 x 24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0), dilation=1),    # 256 x 4 x 24 -> 256 x 1 x 24
            nn.LeakyReLU(),
        )
        
        self.reduced_face_emb_len = 16
        self.emotion_len = 7
        self.E = self.reduced_face_emb_len + self.emotion_len
        # self.E = self.emotion_len
        self.articulation_layers = nn.ModuleList([
            nn.Conv2d(in_channels=256 + self.E,out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1), # (256+E) x 1 x 24 -> (256) x 1 x 13
            nn.Conv2d(in_channels=256 + self.E,out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2), dilation=1), # (256+E) x 1 x 13 -> (256) x 1 x 8
            nn.Conv2d(in_channels=256 + self.E,out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1), # (256+E) x 1 x 8 -> (256) x 1 x 4
            nn.Conv2d(in_channels=256 + self.E,out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1), # (256+E) x 1 x 4 -> (256) x 1 x 2
            nn.Conv2d(in_channels=256 + self.E,out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), dilation=1) # (256+E) x 1 x 2 -> (256) x 1 x 1
        ])
        
        self.LeakyReLU = nn.LeakyReLU()

        self.fc1 =  nn.Linear(256 + self.E, 150)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(150, 478*3)
        self.face_emb_fc = nn.Linear(512, self.reduced_face_emb_len)

        self.init_fc_layers()        

    
    def init_fc_layers(self):
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal_(self.face_emb_fc.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.zeros_(self.face_emb_fc.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, spec, face_emb=None, emotion=None):
        # input : 
        #   spec = B x 2 x 256 x 24 
        #   face_emb = B x 512
        #   emotion = B x 7 
        # output : geometry ( B x 478 x 3)

        reduced_face_emb = self.LeakyReLU(self.face_emb_fc(face_emb))
        out = self.formantAnalysis(spec)
        
        for layer in self.articulation_layers:
            B, _, H, W = out.shape
            expanded_reduced_face_emb = reduced_face_emb.view(B, self.reduced_face_emb_len, 1, 1).expand(B, self.reduced_face_emb_len, H, W)
            expanded_emotion = emotion.view(B, self.emotion_len, 1, 1).expand(B, self.emotion_len, H, W)
            # E = expanded_reduced_face_emb 
            E = torch.cat((expanded_reduced_face_emb, expanded_emotion), dim=1).view(B, self.E, H, W)
            # E = expanded_emotion
            out = layer(torch.cat((out, E), dim=1))
            out = self.LeakyReLU(out)

        B, _, H, W = out.shape
        expanded_reduced_face_emb = reduced_face_emb.view(B, self.reduced_face_emb_len, 1, 1).expand(B, self.reduced_face_emb_len, H, W)
        expanded_emotion = emotion.view(B, self.emotion_len, 1, 1).expand(B, self.emotion_len, H, W)
        # E = expanded_reduced_face_emb 
        E = torch.cat((expanded_reduced_face_emb, expanded_emotion), dim=1).view(B, self.E, H, W)
        # E = expanded_emotion
        out = torch.cat((out, E), dim=1)
        out = out.view(B, -1)
        geometry = self.fc2(self.tanh(self.fc1(out))).view(-1, 478, 3)

        return geometry