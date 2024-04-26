import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets
from torchvision import transforms
from model import (
    DirecEncoder,
    RateEncoder,
    LatencyEncoder,
    ShallowEncoder,
    ShallowDecoder,
    ShallowEncoderRepat,
    ShallowEncoderGLIF,
    LSTMEncoder,
    SimpleNet,
)
from spikingjelly.activation_based.model import (
    sew_resnet,
    spiking_vgg,
    spiking_resnet,
    spiking_vggws_ottt,
    parametric_lif_net,
)
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.surrogate import ATan
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import layer


dataset_dict = {
    "cifar10": datasets.CIFAR10,
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
}

encoder_dict = {
    "rate": RateEncoder,
    "latency": LatencyEncoder,
    "direct": DirecEncoder,
    "shallow": ShallowEncoder,
    "shallow_repeat": ShallowEncoderRepat,
    "shallow_glif": ShallowEncoderGLIF,
    "lstm": LSTMEncoder,
}

decoder_dict = {
    "sew_resnet": sew_resnet.sew_resnet34,
    "spiking_vgg": spiking_vggws_ottt.ottt_spiking_vgg11,
    "spiking_resnet": spiking_resnet.spiking_resnet18,
    "shallow": ShallowDecoder,
    "plif_cifar": parametric_lif_net.CIFAR10Net,
    "plif_mnist": parametric_lif_net.MNISTNet,
    "plif_fmnist": parametric_lif_net.FashionMNISTNet,
    "simplenet": SimpleNet,
}


def get_encoder(cfg: DictConfig):
    encoder = encoder_dict[cfg.encoder.name]
    kwargs = OmegaConf.to_container(cfg.encoder, resolve=True)
    del kwargs["name"]
    if hasattr(cfg.encoder, "path"):
        del kwargs["path"]
    encoder_net = encoder(**kwargs)
    if hasattr(cfg.encoder, "path") and cfg.encoder.path is not None:
        encoder_net.load_state_dict(torch.load(cfg.encoder.path))
        # free encoder net
        for param in encoder_net.parameters():
            param.requires_grad = False

    functional.set_step_mode(encoder_net, "m")
    functional.set_backend(encoder_net, "torch")
    return encoder_net


def get_decoder(cfg: DictConfig):
    decoder = decoder_dict[cfg.decoder.name]
    kwargs = OmegaConf.to_container(cfg.decoder, resolve=True)
    del kwargs["name"]
    if not "plif" in cfg.decoder.name:
        decoder_net = decoder(
            num_classes=cfg.dataset.num_classes,
            spiking_neuron=LIFNode,
            surrogate_function=ATan(),
            **kwargs
        )

    else:
        decoder_net = decoder(
            spiking_neuron=LIFNode, surrogate_function=ATan(), **kwargs
        )

    if cfg.dataset.channels != 3 or hasattr(cfg, "in_channels"):
        if not "vgg" in cfg.decoder.name and not "plif" in cfg.decoder.name:
            in_channels = (
                cfg.dataset.channels if cfg.in_channels is None else cfg.in_channels
            )
            decoder_net.conv1 = layer.Conv2d(
                in_channels,
                decoder_net.conv1.out_channels,
                kernel_size=decoder_net.conv1.kernel_size,
                stride=decoder_net.conv1.stride,
                padding=decoder_net.conv1.padding,
                bias=decoder_net.conv1.bias is not None,
            )
        elif "vgg" in cfg.decoder.name:
            in_channels = (
                cfg.dataset.channels if cfg.in_channels is None else cfg.in_channels
            )
            decoder_net.features[0] = layer.Conv2d(
                in_channels,
                decoder_net.features[0].out_channels,
                kernel_size=decoder_net.features[0].kernel_size,
                stride=decoder_net.features[0].stride,
                groups=decoder_net.features[0].groups,
                padding=decoder_net.features[0].padding,
                bias=decoder_net.features[0].bias is not None,
            )

    functional.set_step_mode(decoder_net, "m")
    functional.set_backend(decoder_net, "torch")
    return decoder_net


def get_autodecoder(cfg: DictConfig):
    decoder = decoder_dict[cfg.decoder.name]
    kwargs = OmegaConf.to_container(cfg.decoder, resolve=True)
    del kwargs["name"]
    decoder_net = decoder(
        in_channels=cfg.in_channels,
        out_channels=cfg.dataset.channels,
        T=cfg.encoder.T,
        **kwargs
    )
    functional.set_step_mode(decoder_net, "m")
    functional.set_backend(decoder_net, "torch")

    return decoder_net


def get_dataset(cfg: DictConfig):
    dataset = dataset_dict[cfg.dataset.name]
    train_transform, test_transform = _get_transform(cfg)
    train_data = dataset(
        root=cfg.dataset.path, train=True, download=True, transform=train_transform
    )
    val_data = dataset(
        root=cfg.dataset.path, train=False, download=True, transform=test_transform
    )
    return train_data, val_data


def _get_transform(cfg: DictConfig):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(cfg.dataset.size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std),
        ]
    )

    return train_transform, test_transform
