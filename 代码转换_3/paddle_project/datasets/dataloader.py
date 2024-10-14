import os
import paddle
import glob
import librosa
from utils.audio import Audio


def create_dataloader(hp, args, train):

    def train_collate_fn(batch):
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()
        for dvec_mel, target_mag, mixed_mag in batch:
            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = paddle.stack(x=target_mag_list, axis=0)
        mixed_mag_list = paddle.stack(x=mixed_mag_list, axis=0)
        return dvec_list, target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch
    if train:
        return paddle.io.DataLoader(dataset=VFDataset(hp, args, True),
            batch_size=hp.train.batch_size, shuffle=True, num_workers=hp.
            train.num_workers, collate_fn=train_collate_fn, drop_last=True)
    else:
        return paddle.io.DataLoader(dataset=VFDataset(hp, args, False),
            collate_fn=test_collate_fn, batch_size=1, shuffle=False,
            num_workers=0)


class VFDataset(paddle.io.Dataset):

    def __init__(self, hp, args, train):

        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir
        self.dvec_list = find_all(hp.form.dvec)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        self.target_mag_list = find_all(hp.form.target.mag)
        self.mixed_mag_list = find_all(hp.form.mixed.mag)
        assert len(self.dvec_list) == len(self.target_wav_list) == len(self
            .mixed_wav_list) == len(self.target_mag_list) == len(self.
            mixed_mag_list), 'number of training files must match'
        assert len(self.dvec_list) != 0, 'no training file found'
        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()
        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = paddle.to_tensor(data=dvec_mel).astype(dtype='float32')
        if self.train:
            target_mag = paddle.load(path=str(self.target_mag_list[idx]))
            mixed_mag = paddle.load(path=str(self.mixed_mag_list[idx]))
            return dvec_mel, target_mag, mixed_mag
        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], self.hp
                .audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], self.hp.
                audio.sample_rate)
            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx]
                )
            target_mag = paddle.to_tensor(data=target_mag)
            mixed_mag = paddle.to_tensor(data=mixed_mag)
            return (dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag,
                mixed_phase)

    def wav2magphase(self, path):
        wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
