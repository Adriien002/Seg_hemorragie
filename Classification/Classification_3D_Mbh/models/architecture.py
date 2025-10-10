from monai.networks.nets import ResNet
import config



def resnet():
    return ResNet(
        block='basic',
        layers=[1, 1, 1, 1],        # Beaucoup moins de couches (vs [2,2,2,2])
        block_inplanes=[32, 64, 128, 256],  # Moins de channels (vs [64,128,256,512])
        spatial_dims=3,
        n_input_channels=1,
        num_classes=config.NUM_CLASSES,
        conv1_t_size=7,
        conv1_t_stride=(2, 2, 2)  )  # Stride dans les 3 dimensions