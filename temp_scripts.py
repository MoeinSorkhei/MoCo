import torch
import torchvision.models as models

import helper


def get_pretrained_model(arch, out_features, keep_mlp, checkpoint_file):
    """
    Creating model and loading the encoder weights from the checkpoint is done in exactly the same procedure as the moco experiments.
    :param arch:
    :param out_features:
    :param keep_mlp:
    :param checkpoint_file:
    :return:
    """
    # create model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()  # arch like 'resnet34'

    # load from pre-trained, before DistributedDataParallel constructor
    print("=> loading checkpoint from '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location="cpu")  # load moco checkpoint into CPU

    # rename moco pre-trained keys. This loads the encoder params into the created resnet params
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # add new keys with the prefix removed 'module.encoder_q.conv1.weight' -> 'conv1.weight' so it they can be loaded into a resnet model
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded encoder weights of pre-trained model from: '{}'".format(checkpoint_file))

    # replace fc layer with a new one for the new task
    model.fc = torch.nn.Linear(in_features=512, out_features=out_features)
    return model.cuda()


def investigate_moco():
    helper.set_gpu_devices(devices='7')
    pretrained_resnet = get_pretrained_model(arch='resnet34', out_features=8, keep_mlp=False, checkpoint_file='/storage/sorkhei/MoCo/checkpoint_0100.pth.tar')

    t = torch.randn((1, 3, 312, 256)).cuda()
    print(pretrained_resnet(t).shape)


def main():
    investigate_moco()


if __name__ == '__main__':
    main()
