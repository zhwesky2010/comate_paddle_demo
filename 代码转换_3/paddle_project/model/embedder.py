import paddle


class LinearNorm(paddle.nn.Layer):

    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = paddle.nn.Linear(in_features=hp.embedder.
            lstm_hidden, out_features=hp.embedder.emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(paddle.nn.Layer):

    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = paddle.nn.LSTM(input_size=hp.embedder.num_mels,
            hidden_size=hp.embedder.lstm_hidden, num_layers=hp.embedder.
            lstm_layers, time_major=not True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        mels = mel.unfold(axis=1, size=self.hp.embedder.window, step=self.
            hp.embedder.stride)
        mels = mels.transpose(perm=[1, 2, 0])
        x, _ = self.lstm(mels)
        x = x[:, -1, :]
        x = self.proj(x)
        x = x / paddle.linalg.norm(x=x, p=2, axis=1, keepdim=True)
        x = x.sum(axis=0) / x.shape[0]
        return x
