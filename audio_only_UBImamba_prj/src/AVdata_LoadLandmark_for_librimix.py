#TODO：数据集数据加载
# Created on 2022/05
# Author: de bang liu
# Refer to Kaituo XU  in https://github.com/kaituoxu/Conv-TasNet
import json
import math
import os
import numpy as np
import torch
import torch.utils.data as data
import librosa
import pandas as pd

fps_org = 25

class AudioandVideoDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioandVideoDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix_clean.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')

        # mix_json = os.path.join(json_dir, 'mixed.json')
        # s1_json = os.path.join(json_dir, 'speaker1.json')
        # s2_json = os.path.join(json_dir, 'speaker2.json')

        with open(mix_json, 'r') as f:
            sorted_mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            sorted_s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            sorted_s2_infos = json.load(f)

        #TODO visual data
        # s1Video_json = os.path.join(json_dir, 'speaker1VideomouthYLandMark.json')
        # s2Video_json = os.path.join(json_dir, 'speaker2VideomouthYLandMark.json')



# #for VOx test
#         if int(len(sorted_mix_infos)) > 20000:
#             framstrat = int(len(sorted_mix_infos) * 0.40)
#             framlength = int(len(sorted_mix_infos) * 0.44)
#             print(framstrat, framlength)#train
#         else:
#             framstrat = int(len(sorted_mix_infos) * 0.0)
#             framlength = int(len(sorted_mix_infos) * 0.1)
#             print(framstrat, framlength)#val
#         sorted_mix_infos = sorted_mix_infos[framstrat:framlength]
#         sorted_s1_infos = sorted_s1_infos[framstrat:framlength]
#         sorted_s2_infos = sorted_s2_infos[framstrat:framlength]
#         s1Video_infos = s1Video_infos[framstrat:framlength]
#         s2Video_infos = s2Video_infos[framstrat:framlength]
#for Vox/ GRID/LRS2
        framstrat = int(len(sorted_mix_infos) * 0.0)
        framlength = int(len(sorted_mix_infos) * 1.0)
        print(framstrat, framlength)
        sorted_mix_infos = sorted_mix_infos[framstrat:framlength]
        sorted_s1_infos = sorted_s1_infos[framstrat:framlength]
        sorted_s2_infos = sorted_s2_infos[framstrat:framlength]




        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            video_frame_len = int(segment * fps_org)
            drop_utt, drop_len = 0, 0
            for i,(_,sample) in enumerate(sorted_mix_infos):

                _, s1_frame = sorted_s1_infos[i]
                _, s2_frame = sorted_s2_infos[i]

                if  (sample < segment_len) or (s1_frame < segment_len) or (s2_frame < segment_len) :
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples or Audio visual frame not alignment ".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []

                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    s1_utt_len = int(sorted_s1_infos[end][1])
                    s2_utt_len = int(sorted_s2_infos[end][1])

                    if   (utt_len >= segment_len) and (s1_utt_len >= segment_len) and (s2_utt_len >= segment_len) :  # skip too short utt or not not alignment AV frame

                        num_segments += 1#只取一个utts
                        if num_segments > batch_size:
                            if start == end: end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])

                        assert len(part_mix) == len(part_s1) == len(part_s2)

                    end += 1

                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2,
                                      sample_rate, segment_len,video_frame_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch


    def __getitem__(self, index):
        # print(self.minibatch[index])
        return self.minibatch[index]

    def __len__(self):
        # print(len(self.minibatch))
        return len(self.minibatch)


class AudioandVideoDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioandVideoDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    # print('aaaaaa')
    # print("aaaaaa",batch)
    assert len(batch) == 1
    mixtures, sources = load_mixtures_and_sources(batch[0])
    # print('mixtures.size()',len(mixtures))
    # print('sources.size()',len(sources))
    # print('video_list.size()',len(video_list))
    # print('mixtures.size()',mixtures[0].shape)
    # print('sources.size()',sources[0].shape)
    # print('video_list.size()',video_list[0].shape)

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()


    # print(video_list_pad.size())
    # print('ilens.size()',ilens.size())
    # print('mixtures_pad.size()',mixtures_pad.size())
    # print('sources_pad.size()',sources_pad.size())

    return mixtures_pad, ilens, sources_pad


# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources ,video_list= [], [], []
    mix_infos, s1_infos, s2_infos,sample_rate, segment_len,video_frame_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]



        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)



        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        if segment_len >= 0:
            mixtures.append(mix[0:segment_len])
            sources.append(s[0:segment_len])

    return mixtures, sources


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioandVideoDataset(json_dir, int(batch_size))
    data_loader = AudioandVideoDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
