import sys
sys.path.append('/workspace/comate_paddle_demo/代码转换_3/paddle_project/utils')
import paddle_aux
import paddle


class VoiceFilter(paddle.nn.Layer):

    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, 'stft-related dimension mismatch'
        self.conv = paddle.nn.Sequential(paddle.nn.ZeroPad2D(padding=(3, 3,
            0, 0)), paddle.nn.Conv2D(in_channels=1, out_channels=64,
            kernel_size=(1, 7), dilation=(1, 1)), paddle.nn.BatchNorm2D(
            num_features=64), paddle.nn.ReLU(), paddle.nn.ZeroPad2D(padding
            =(0, 0, 3, 3)), paddle.nn.Conv2D(in_channels=64, out_channels=
            64, kernel_size=(7, 1), dilation=(1, 1)), paddle.nn.BatchNorm2D
            (num_features=64), paddle.nn.ReLU(), paddle.nn.ZeroPad2D(
            padding=2), paddle.nn.Conv2D(in_channels=64, out_channels=64,
            kernel_size=(5, 5), dilation=(1, 1)), paddle.nn.BatchNorm2D(
            num_features=64), paddle.nn.ReLU(), paddle.nn.ZeroPad2D(padding
            =(2, 2, 4, 4)), paddle.nn.Conv2D(in_channels=64, out_channels=
            64, kernel_size=(5, 5), dilation=(2, 1)), paddle.nn.BatchNorm2D
            (num_features=64), paddle.nn.ReLU(), paddle.nn.ZeroPad2D(
            padding=(2, 2, 8, 8)), paddle.nn.Conv2D(in_channels=64,
            out_channels=64, kernel_size=(5, 5), dilation=(4, 1)), paddle.
            nn.BatchNorm2D(num_features=64), paddle.nn.ReLU(), paddle.nn.
            ZeroPad2D(padding=(2, 2, 16, 16)), paddle.nn.Conv2D(in_channels
            =64, out_channels=64, kernel_size=(5, 5), dilation=(8, 1)),
            paddle.nn.BatchNorm2D(num_features=64), paddle.nn.ReLU(),
            paddle.nn.ZeroPad2D(padding=(2, 2, 32, 32)), paddle.nn.Conv2D(
            in_channels=64, out_channels=64, kernel_size=(5, 5), dilation=(
            16, 1)), paddle.nn.BatchNorm2D(num_features=64), paddle.nn.ReLU
            (), paddle.nn.Conv2D(in_channels=64, out_channels=8,
            kernel_size=(1, 1), dilation=(1, 1)), paddle.nn.BatchNorm2D(
            num_features=8), paddle.nn.ReLU())
        self.lstm = paddle.nn.LSTM(input_size=8 * hp.audio.num_freq + hp.
            embedder.emb_dim, hidden_size=hp.model.lstm_dim, time_major=not
            True, direction='bidirect')
        self.fc1 = paddle.nn.Linear(in_features=2 * hp.model.lstm_dim,
            out_features=hp.model.fc1_dim)
        self.fc2 = paddle.nn.Linear(in_features=hp.model.fc1_dim,
            out_features=hp.model.fc2_dim)

    def forward(self, x, dvec):
        x = x.unsqueeze(axis=1)
        x = self.conv(x)
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2)
            ).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        dvec = dvec.unsqueeze(axis=1)
        dvec = dvec.tile(repeat_times=[1, x.shape[1], 1])
        x = paddle.concat(x=(x, dvec), axis=2)
        x, _ = self.lstm(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.fc2(x)
        x = paddle.nn.functional.sigmoid(x=x)
        return x
