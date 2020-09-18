from torch import nn

class RnnNet(nn.Module):
    def __init__(self):
        super(RnnNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128), # NV N为NS（s为验证码的W）
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )

        self.lstm = nn.LSTM(128, 246, 2, batch_first=True)  # 128个输入，246个隐藏层，2层神经元
        self.lstm2 = nn.LSTM(246, 246, 2, batch_first=True)
        # self.lstm = nn.LSTM(128, 128, 2, batch_first=True)
        self.out = nn.Linear(246, 10) # 用来做十个数字的分类任务
    def forward(self, input): # 此时输入为NCHW（N，3，60，120）
        input = input.reshape(-1, 180, 120).permute(0,2,1) # 将NCHW->N180W->NW180 （将图片竖着切分传入网络）
        input = input.reshape(-1, 180) # NW180->N180 (输入全连接的结构)
        out_fc1 = self.fc1(input) # N180->N128 N为NS（s为验证码的W）
        out_fc1 = out_fc1.reshape(-1, 120, 128) # 拆分为NSV的结构，传入LSTM
        out_lstm, (hn, cn) = self.lstm(out_fc1) #out_lstm为NSV结构
        out = out_lstm[:,-1,:] # 取最后一个S，包含整个图片的特征 N246

        out = out.reshape(-1, 1, 246) # NV->N1V
        out = out.expand(-1, 4, 246) # N1V->N4V

        out_lstm1,(hn,cn) = self.lstm2(out) # NSV 4,246

        out_lstm1 = out_lstm1.reshape(-1, 246) # N,4,246->N*4,246
        out_fc2 = self.out(out_lstm1) # N*4,10
        out = out_fc2.reshape(-1, 4, 10) # N*4,10->N,4,10
        return out

