#TODO：Audio_only train
# Created on 2023/12
# Author: de bang liu


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch

# from data import AudioDataLoader, AudioDataset
# from AVdata_LoadLandmark1 import AudioandVideoDataLoader, AudioandVideoDataset
from  AVdata_LoadLandmark_for_librimix  import AudioandVideoDataLoader, AudioandVideoDataset

# from AVdata_LoadLandmark_for_lrs3 import AudioandVideoDataLoader, AudioandVideoDataset
# from AVdata_LoadLandmark_for_Vox2  import AudioandVideoDataLoader, AudioandVideoDataset# # from lrs2_AVdataloader import mixtures_and_sources_loader, Audio_and_Video_Dataset_Drop,two_Videodata_loader,Videodata_loader

# from lrs2_landmark_AVdataloader import mixtures_and_sources_loader, AudioDataset,VideoDataset,land_mark_loader,land_mark_twospeaker_loader
# from AVdataloader import AudioDataLoader, AudioDataset

# from audio_only_conv_tasnet import ConvTasNet
# from audio_only_tasnet import TasNet
# from dptnet import DPTNet
# from AFRCNNsum import AFRCNNsum
from AFRCNN import AFRCNN
# from audio_only_Dualpathrnn import Dual_RNN_model
# from SuDORMRF import SuDORMRF
# from asteroid.models import SuDORMRFNet
# from Sepformer_Wrapper import SepformerWrapper
# from  audio_only_Ucoder_DPRNN import audio_only_Ucoder_DPRNN
# from Audio_only_solver_AMP_new import Solver
# from Audio_only_solver import Solver
# from Audio_visual_solver_for_audio_only import Solver
from Audio_only_solver_for_audio_only import  Solver

from asteroid import  DPRNNTasNet
from asteroid import ConvTasNet
# from RE_Sepformer import ReSepformer
# from MoreEfficientUBImamba_SS05_new_res2_avgpool_v3_silu_v2_mambaconfigv1_16_4_2_5layer_share_v1_2_nofusmamba import MoreEfficientUBImamba_SS05_new_res2_avgpool_v3_silu_v2_mambaconfigv1_16_4_2_5layer_share_v1_2_nofusmamba
#



parser = argparse.ArgumentParser("Selective Structured State Space Model with Temporal Dilated Convolution for Efficient Speech Separation (Speech-Conv-Mamba) "
                                                                  "with Permutation Invariant Training")


train_dir_GRID = '/media/debangliu/datapart_ssd/dataset/GRID/jsonfiles_for_ubuntu/Max8k_AllType_landmark/tr'
valid_dir_GRID = '/media/debangliu/datapart_ssd/dataset/GRID/jsonfiles_for_ubuntu/Max8k_AllType_landmark/cv'



train_dir_librimix = '/media/debangliu/datapart_ssd/librimix_nonoise/jsonfiles_for_ubuntu/min/train-100'
valid_dir_librimix = '/media/debangliu/datapart_ssd/librimix_nonoise/jsonfiles_for_ubuntu/min/dev'


parser.add_argument('--train_dir', type=str, default=train_dir_librimix,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=valid_dir_librimix,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
#TODO: segment  default=2
parser.add_argument('--segment', default=2, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=512, type=int,
                    help='Number of filters in autoencoder')
#TODO: '--L'  default=40
parser.add_argument('--L', default=16, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=128, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=3, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
#TODO: use_cuda  default=0 cpu模式  default =1  GPU模式
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
#TODO：epochs
parser.add_argument('--epochs', default=150, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')

#TODO:batch_size
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
#TODO:learn rate
parser.add_argument('--lr', default=2e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--save_logfolder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=100, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    tr_dataset = AudioandVideoDataset(args.train_dir, args.batch_size,
                                      sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset = AudioandVideoDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                                      sample_rate=args.sample_rate, #todo can not  set segment to -1
                                      segment=args.segment, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    tr_loader = AudioandVideoDataLoader(tr_dataset, batch_size=1,
                                        shuffle=args.shuffle,
                                        num_workers=args.num_workers)
    cv_loader = AudioandVideoDataLoader(cv_dataset, batch_size=1,
                                        num_workers=0)

    ##################################################


    data =  {'tr_loader': tr_loader,'cv_loader':cv_loader}

    # print(args.N, args.L, args.B, args.H, args.P, args.X, args.R, args.C,) #512 16 128 512 3 8 3 2
    # model
    # model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    #                    args.C, norm_type=args.norm_type, causal=args.causal,
    #                    mask_nonlinear=args.mask_nonlinear)

    # model = ConvTasNet(n_src=2,kernel_size=16,stride=8)
    # model = ConvTasNet(n_src=2,kernel_size=10,stride=5)

    # model = Dual_RNN_model(256, 64, 128, kernel_size=2, bidirectional=True, norm='ln', num_layers=6,K=150)

    # model = DPRNNTasNet(n_src=2,kernel_size=8,stride=4,n_filters=64,bn_chan=64,hid_size=128,chunk_size=150)
    # model = DPRNNTasNet(n_src=2,kernel_size=4,stride=2,n_filters=64,bn_chan=64,hid_size=128,chunk_size=200)
    # model = DPRNNTasNet(n_src=2,kernel_size=2,stride=1,n_filters=64,bn_chan=64,hid_size=128,chunk_size=250)

    # model = Dual_RNN_model(64, 64, 128,kernel_size=8,bidirectional=True, norm='ln', num_layers=6,K=150)
    # model = Dual_RNN_model(64, 64, 128,kernel_size=4,bidirectional=True, norm='ln', num_layers=6,K=200)
    # model = Dual_RNN_model(64, 64, 128,kernel_size=2,bidirectional=True, norm='ln', num_layers=6,K=250)
    # model = audio_only_Ucoder_DPRNN(160, 64, 128, kernel_size=40, bidirectional=True,  norm='ln', segment =2,samplerate=8000,VitRGBSize=(40,40))
    # model = DPTNet(N=64, C=2, L=2, H=4, K=250, B=6)
    # model = AFRCNNsum(out_channels=512, in_channels=512, num_blocks=16,upsampling_depth=5,enc_kernel_size=21, enc_num_basis=512, num_sources=2)
    model = AFRCNN(out_channels=512, in_channels=512, num_blocks=8,upsampling_depth=5,enc_kernel_size=21, enc_num_basis=512, num_sources=2)

    # model = AFRCNN.load_model(   '/data/debangliu/runpart/speechseparation/audio_only_audiovisual/src/exp/temp/final.pth.tar')

    # model = SuDORMRF(out_channels=128,in_channels=512,  num_blocks=40,upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512,num_sources=2)
    # model = SuDORMRFNet(
    #     n_src=2,
    #     n_filters=512,
    #     fb_name="free",
    #     kernel_size=21,
    #     stride=10,
    #     bn_chan=128,
    #     num_blocks=40,
    #     upsampling_depth=4,
    #     mask_act="softmax",
    #     in_chan=512)
    # model = SepformerWrapper(encoder_kernel_size=20,
    #                          encoder_in_nchannels=1,
    #                          encoder_out_nchannels=256,
    #                          masknet_chunksize=250,
    #                          masknet_numlayers=2,
    #                          masknet_norm="ln",
    #                          masknet_useextralinearlayer=False,
    #                          masknet_extraskipconnection=True,
    #                          masknet_numspks=2,
    #                          intra_numlayers=8,
    #                          inter_numlayers=8,
    #                          intra_nhead=8,
    #                          inter_nhead=8,
    #                          intra_dffn=1024,
    #                          inter_dffn=1024,
    #                          intra_use_positional=True,
    #                          inter_use_positional=True,
    #                          intra_norm_before=True,
    #                          inter_norm_before=True, )
    # model = U2net_wave_14_NoSeqLearningNet_new(in_ch=1,out_ch=1,in_channels=96)
    # model = U2net_wave_14_4_4_2(in_ch=1,out_ch=1,in_channels=128)
    # model = ReSepformer(encoder_kernel_size=16,
    #     encoder_in_nchannels=1,
    #     encoder_out_nchannels=128,
    #     masknet_numspks=2,)
    # model = U2net_wave_14_4_4_final_4(in_ch=1,out_ch=1,in_channels=64)
    # model = MoreEfficientMLED_04(in_ch=1, out_ch=1, in_channels=88)

    # model = MoreEfficientUBImamba_SS05_new_res2_avgpool_v3_silu_v2_mambaconfigv1_16_4_2_5layer_share_v1_2_nofusmamba(in_ch=1,out_ch=1,in_channels=88,mid_ch=32,dilation_layer=6,net_layer=1)


    print(model.__class__.__name__)
    filename = open('train_log.txt', 'w')
    filename.write("data set path:%s'\n'%s" %(args.train_dir,args.valid_dir))
    filename.write('\n')
    filename.write('\n')
    filename.write(str(model))
    filename.close()

    # ######################################
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # ######################################


    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
        # torch.backends.cudnn.benchmark = True
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    # elif args.optimizer == 'adam':
    #     optimizier = torch.optim.Adam(model.parameters(),
    #                                   lr=args.lr,
    #                                   weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizier, 2, gamma=0.98, last_epoch=-1)
    else:
        print("Not support optimizer")
        return

        # solver

    solver = Solver(data, model, optimizier, args, scheduler)
    solver.train()




if __name__ == '__main__':
    args = parser.parse_args()
    # print(torch.cuda.is_available())
    print(args)
    main(args)

    # evaluate(args)

