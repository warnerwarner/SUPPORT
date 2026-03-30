import torch
import torch.nn as nn

from model.convhole import ConvHole2D


class SUPPORT(nn.Module):
    """
    Blindspot network

    Arguments:
        in_channels: the number of input channels (int)
        mid_channels: the number of middle channels ([int])
    """

    def __init__(
        self,
        in_channels,
        mid_channels=[16, 32, 64, 128, 256],
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=[64, 32, 16],
        bs_size=1,
        bp=False,
        is_raw=False,
        prevent_injection=False,
        use_phase_conditioning=False,
    ):
        super(SUPPORT, self).__init__()

        # check arguments
        if len(mid_channels) < 2:
            raise Exception("length of mid_channels must be larger than 1")
        # if depth % 2 == 0:
        #     raise Exception("depth must be an odd number")
        if type(blind_conv_channels) != int:
            raise Exception("type of blind_conv_channels must be an integer")
        if not all([type(i) == int for i in one_by_one_channels]):
            raise Exception("one_by_one_channels must be an integer array")

        self.in_channels = in_channels
        self.out_channels = 1
        self.mid_channels = mid_channels
        self.depth = depth
        self.depth3x3 = depth
        self.depth5x5 = depth - 2
        self.prevent_injection = prevent_injection

        self.blind_conv_channels = blind_conv_channels
        self.one_by_one_channels = one_by_one_channels

        self.last_layer_channels = last_layer_channels

        if type(bs_size) == int:
            bs_size = [bs_size, bs_size]
        self.bs_size = bs_size

        self.bp = bp
        self.is_raw = is_raw
        self.use_phase_conditioning = use_phase_conditioning

        if in_channels == 1:
            self.twod = True
        else:
            self.twod = False

        assert not (
            self.bp and self.twod
        ), "two options cannot be selected in same time."

        # initialize
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.maxpool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_2d = nn.Upsample(scale_factor=2)

        if self.twod is False:
            self._gen_unet()
        if bp is False:
            self._gen_bsnet()

        # last layer
        last_layers = []
        for idx, c in enumerate(self.last_layer_channels):
            if idx == 0:
                if bp is False and self.twod is False:
                    last_layers.append(
                        nn.Conv2d(
                            2 * self.one_by_one_channels[-1],
                            c,
                            kernel_size=1,
                            padding=0,
                        )
                    )
                else:
                    last_layers.append(
                        nn.Conv2d(
                            self.one_by_one_channels[-1], c, kernel_size=1, padding=0
                        )
                    )
            else:
                last_layers.append(
                    nn.Conv2d(last_layer_channels[idx - 1], c, kernel_size=1, padding=0)
                )
            last_layers.append(nn.BatchNorm2d(c))
            last_layers.append(self.leaky_relu)

        last_layers.append(nn.Conv2d(c, self.out_channels, kernel_size=1, padding=0))

        self.last_layers = nn.ModuleList(last_layers)

    def _gen_unet(self):
        # (Unet) encoding layers
        # When phase conditioning is enabled, we concatenate sin+cos phase channels
        # for each non-center frame: (T-1) image + (T-1) sin + (T-1) cos = 3*(T-1)
        unet_in_channels = self.in_channels - 1
        if self.use_phase_conditioning:
            unet_in_channels = 3 * (self.in_channels - 1)

        self.enc_layers = []
        self.enc_bn_layers = []
        for i in range(len(self.mid_channels)):
            if i == 0:
                self.enc_layers.append(
                    nn.Conv2d(
                        unet_in_channels,
                        self.mid_channels[i],
                        kernel_size=3,
                        padding=1,
                    )
                )
            else:
                self.enc_layers.append(
                    nn.Conv2d(
                        self.mid_channels[i - 1],
                        self.mid_channels[i],
                        kernel_size=3,
                        padding=1,
                    )
                )
            self.enc_bn_layers.append(nn.BatchNorm2d(self.mid_channels[i]))
        self.enc_layers = nn.ModuleList(self.enc_layers)
        self.enc_bn_layers = nn.ModuleList(self.enc_bn_layers)
        self.enc_bn_layers = nn.ModuleList(self.enc_bn_layers)

        # (Unet) decoding layers
        self.dec_layers = []
        self.dec_bn_layers = []
        for i in range(len(self.mid_channels) - 1):
            self.dec_layers.append(
                nn.Conv2d(
                    self.mid_channels[i] + self.mid_channels[i + 1],
                    self.mid_channels[i],
                    kernel_size=3,
                    padding=1,
                )
            )
            self.dec_bn_layers.append(nn.BatchNorm2d(self.mid_channels[i]))
        self.dec_layers = nn.ModuleList(reversed(self.dec_layers))
        self.dec_bn_layers = nn.ModuleList(reversed(self.dec_bn_layers))

        # (Unet) 1x1 convs
        self.unet_1_convs = []
        self.unet_1_convs_bn = []
        for idx, c in enumerate(self.one_by_one_channels):
            if idx == 0:
                self.unet_1_convs.append(
                    nn.Conv2d(self.mid_channels[0], c, kernel_size=1, padding=0)
                )
            else:
                self.unet_1_convs.append(
                    nn.Conv2d(
                        self.one_by_one_channels[idx - 1], c, kernel_size=1, padding=0
                    )
                )
            self.unet_1_convs_bn.append(nn.BatchNorm2d(c))
        self.unet_1_convs = nn.ModuleList(self.unet_1_convs)
        self.unet_1_convs_bn = nn.ModuleList(self.unet_1_convs_bn)

    def _gen_bsnet(self):
        # (BS) additional parameters
        self.scalars_3x3 = []
        # c_in_first = self.one_by_one_channels[-1] + 1
        for d in range(self.depth3x3 - 1):
            c_in = 1 if d == 0 else self.blind_conv_channels
            self.scalars_3x3.append(nn.Parameter(torch.ones(c_in), requires_grad=True))
        self.scalars_3x3 = nn.ParameterList(self.scalars_3x3)

        self.scalars_5x5 = []
        for d in range(self.depth5x5 - 1):
            c_in = 1 if d == 0 else self.blind_conv_channels
            self.scalars_5x5.append(nn.Parameter(torch.ones(c_in), requires_grad=True))
        self.scalars_5x5 = nn.ParameterList(self.scalars_5x5)

        # Per-layer 1x1 convs to project phase-augmented input (3 ch) into
        # blind_conv_channels for re-injection at layers c >= 1.
        # Only created when phase conditioning is active.
        if self.use_phase_conditioning:
            self.inject_proj_3x3 = nn.ModuleList(
                [
                    nn.Conv2d(3, self.blind_conv_channels, kernel_size=1)
                    for _ in range(self.depth3x3 - 1)
                ]
            )
            self.inject_proj_5x5 = nn.ModuleList(
                [
                    nn.Conv2d(3, self.blind_conv_channels, kernel_size=1)
                    for _ in range(self.depth5x5 - 1)
                ]
            )

        # (BS) first layer to process Unet output
        conv3x3 = []
        # Calculate input channels: UNet output + 2 (sin/cos of center-frame row phase)
        bs_in_channels = self.one_by_one_channels[-1]
        if self.use_phase_conditioning:
            bs_in_channels += 2

        conv3x3.append(
            nn.Conv2d(
                bs_in_channels,
                self.blind_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                padding_mode="zeros",
                dilation=1,
            )
        )
        conv3x3.append(nn.BatchNorm2d(self.blind_conv_channels))
        conv3x3.append(self.leaky_relu)
        self.conv3x3 = nn.ModuleList(conv3x3)

        conv5x5 = []
        conv5x5.append(
            nn.Conv2d(
                bs_in_channels,
                self.blind_conv_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
                padding_mode="zeros",
                dilation=1,
            )
        )
        conv5x5.append(nn.BatchNorm2d(self.blind_conv_channels))
        conv5x5.append(self.leaky_relu)
        self.conv5x5 = nn.ModuleList(conv5x5)

        # (BS) dilated convolutions with blind-spot
        blind_conv3x3_layers = []
        for d in range(self.depth3x3):
            if d == 0:
                c_in = 3 if self.use_phase_conditioning else 1
            else:
                c_in = self.blind_conv_channels

            # NOTE: Always use isotropic dilations regardless of is_raw flag
            # The is_raw flag is used for data loading/alignment only
            # Anisotropic dilations were attempted but caused blind spot failure
            # Gold standard training used isotropic dilations even with is_raw=True
            pd = [pow(2, d), pow(2, d)]
            if d == self.depth3x3 - 1:
                pd[0] = pd[0] + self.bs_size[0] // 2
                pd[1] = pd[1] + self.bs_size[1] // 2

            blind_conv3x3_layers.append(
                ConvHole2D(
                    c_in,
                    self.blind_conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=pd,
                    bias=True,
                    padding_mode="zeros",
                    dilation=pd,
                )
            )
            blind_conv3x3_layers.append(nn.BatchNorm2d(self.blind_conv_channels))
            blind_conv3x3_layers.append(self.leaky_relu)
        self.blind_convs3x3 = nn.ModuleList(blind_conv3x3_layers)

        blind_conv5x5_layers = []
        for d in range(self.depth5x5):
            if d == 0:
                c_in = 3 if self.use_phase_conditioning else 1
            else:
                c_in = self.blind_conv_channels

            # NOTE: Always use isotropic dilations regardless of is_raw flag
            # The is_raw flag is used for data loading/alignment only
            pd = [pow(3, d), pow(3, d)]
            if d == self.depth5x5 - 1:
                pd[0] = pd[0] + self.bs_size[0] // 2
                pd[1] = pd[1] + self.bs_size[1] // 2

            blind_conv5x5_layers.append(
                ConvHole2D(
                    c_in,
                    self.blind_conv_channels,
                    kernel_size=5,
                    stride=1,
                    padding=[pd[0] * 2, pd[1] * 2],
                    bias=True,
                    padding_mode="zeros",
                    dilation=pd,
                )
            )
            blind_conv5x5_layers.append(nn.BatchNorm2d(self.blind_conv_channels))
            blind_conv5x5_layers.append(self.leaky_relu)
        self.blind_convs5x5 = nn.ModuleList(blind_conv5x5_layers)

        # (BS) 1x1 convolutions
        out_convs = []
        for idx, c in enumerate(self.one_by_one_channels):
            if self.bs_size[0] == 1 and self.bs_size[1] == 1:
                c_in = (
                    self.blind_conv_channels * (self.depth3x3 + self.depth5x5)
                    if idx == 0
                    else self.one_by_one_channels[idx - 1]
                )
            else:
                c_in = (
                    self.blind_conv_channels * 2
                    if idx == 0
                    else self.one_by_one_channels[idx - 1]
                )
            out_convs.append(
                nn.Conv2d(
                    c_in,
                    c,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )
            out_convs.append(nn.BatchNorm2d(c))
            out_convs.append(self.leaky_relu)
        self.out_convs = nn.ModuleList(out_convs)

    def forward_unet(self, x):
        # x = [b, T, d1, d2]
        # d1, d2 = 512, paper reference
        xs = []
        # print(x.size(), x.min(), x.max(), 'unet')

        for idx, enc_layer in enumerate(self.enc_layers):
            x = self.leaky_relu(self.enc_bn_layers[idx](enc_layer(x)))
            if idx != len(self.enc_layers) - 1:
                xs.append(x)
                x = self.maxpool_2d(x)

        for idx, dec_layer in enumerate(self.dec_layers):
            # up_ = self.upsample_2d(x)
            # print(xs[-idx-1].size(), xs[-idx-1].size()[2:])
            up_ = torch.nn.functional.interpolate(x, xs[-idx - 1].size()[2:])

            x = torch.cat([up_, xs[-idx - 1]], dim=1)
            x = self.leaky_relu(self.dec_bn_layers[idx](dec_layer(x)))

        for idx, one_conv in enumerate(self.unet_1_convs):
            x = self.leaky_relu(self.unet_1_convs_bn[idx](one_conv(x)))

        return x

    def forward_bsnet(self, x, unet_out, phase_sin=None, phase_cos=None):
        # x : bsnet input (B, 1, H, W) — raw center frame
        # unet_out : unet output

        hc = []

        # Build phase-augmented input for injection at every BS-net layer
        # When phase conditioning is active, x_inject = (B, 3, H, W): center frame + sin + cos
        # Otherwise, x_inject = x = (B, 1, H, W)
        if self.use_phase_conditioning and phase_sin is not None:
            B, C, H, W = x.shape
            p_sin = phase_sin.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W)
            p_cos = phase_cos.unsqueeze(1).unsqueeze(-1).expand(B, 1, H, W)
            x_inject = torch.cat([x, p_sin, p_cos], dim=1)  # (B, 3, H, W)
        else:
            x_inject = x  # (B, 1, H, W)

        if unet_out is not None:
            # Concatenate phase info to U-Net output if enabled
            if self.use_phase_conditioning and phase_sin is not None:
                # phase_sin, phase_cos: (B, H) — one phase per row for the center frame
                # Reshape to (B, 1, H, 1) and broadcast to (B, 1, H, W)
                B_u, C_u, H_u, W_u = unet_out.shape
                up_sin = phase_sin.unsqueeze(1).unsqueeze(-1).expand(B_u, 1, H_u, W_u)
                up_cos = phase_cos.unsqueeze(1).unsqueeze(-1).expand(B_u, 1, H_u, W_u)
                # Concatenate along the channel dimension: adds 2 channels
                unet_out = torch.cat([unet_out, up_sin, up_cos], dim=1)

            unet_out1 = self.conv3x3[0](unet_out)
            unet_out1 = self.conv3x3[1](unet_out1)
            unet_out1 = self.conv3x3[2](unet_out1)

        for c in range(self.depth3x3):
            if c == 0:
                x1 = x_inject
            else:
                # x1 = x1 + x1.max() * inp
                # print(x.size())
                if not self.prevent_injection:
                    if self.use_phase_conditioning and phase_sin is not None:
                        # Phase mode: use learned 1x1 conv projection
                        x1 = x1 + self.inject_proj_3x3[c - 1](x_inject)
                    else:
                        # Non-phase mode: use original scalar injection with x (1 channel)
                        x1 = x1 + (
                            self.scalars_3x3[c - 1] * x.permute(0, 2, 3, 1)
                        ).permute(0, 3, 1, 2)
                else:
                    x1 = x1

            x1 = self.blind_convs3x3[3 * c](x1)
            x1 = self.blind_convs3x3[3 * c + 1](x1)
            x1 = self.blind_convs3x3[3 * c + 2](x1)

            if c == 0 and unet_out is not None:
                x1 = x1 + unet_out1

            if self.bs_size[0] == 1 and self.bs_size[1] == 1:
                hc.append(x1)
            else:
                if c == self.depth3x3 - 1:
                    hc.append(x1)

        if unet_out is not None:
            unet_out2 = self.conv5x5[0](unet_out)
            unet_out2 = self.conv5x5[1](unet_out2)
            unet_out2 = self.conv5x5[2](unet_out2)

        for c in range(self.depth5x5):
            if c == 0:
                x2 = x_inject
            else:
                if not self.prevent_injection:
                    if self.use_phase_conditioning and phase_sin is not None:
                        # Phase mode: use learned 1x1 conv projection
                        x2 = x2 + self.inject_proj_5x5[c - 1](x_inject)
                    else:
                        # Non-phase mode: use original scalar injection with x (1 channel)
                        x2 = x2 + (
                            self.scalars_5x5[c - 1] * x.permute(0, 2, 3, 1)
                        ).permute(0, 3, 1, 2)
                else:
                    x2 = x2

            x2 = self.blind_convs5x5[3 * c](x2)
            x2 = self.blind_convs5x5[3 * c + 1](x2)
            x2 = self.blind_convs5x5[3 * c + 2](x2)

            if c == 0 and unet_out is not None:
                x2 = x2 + unet_out2

            if self.bs_size[0] == 1 and self.bs_size[1] == 1:
                hc.append(x2)
            else:
                if c == self.depth5x5 - 1:
                    hc.append(x2)

        x = torch.cat(hc, dim=1)

        for i in range(len(self.out_convs) // 3):
            x = self.out_convs[3 * i + 2](
                self.out_convs[3 * i + 1](self.out_convs[3 * i](x))
            )

        return x

    def forward(self, x, phase_sin=None, phase_cos=None):
        # x = [B, T, H, W]
        # phase_sin, phase_cos = [B, T, H] (per-row phase for all frames in patch)

        center = self.in_channels // 2

        unet_in = torch.cat(
            [
                x[:, :center, :, :],
                x[:, center + 1 :, :, :],
            ],
            dim=1,
        )
        bsnet_in = torch.unsqueeze(x[:, center, :, :], dim=1)

        # Split phase into U-Net portion (non-center frames) and BS-net portion (center frame)
        unet_phase_sin = None
        unet_phase_cos = None
        bsnet_phase_sin = None
        bsnet_phase_cos = None

        if self.use_phase_conditioning and phase_sin is not None:
            # phase_sin/cos: (B, T, H) -> split like the image channels
            # U-Net gets non-center frames: (B, T-1, H)
            unet_phase_sin = torch.cat(
                [phase_sin[:, :center, :], phase_sin[:, center + 1 :, :]], dim=1
            )
            unet_phase_cos = torch.cat(
                [phase_cos[:, :center, :], phase_cos[:, center + 1 :, :]], dim=1
            )
            # BS-net gets center frame only: (B, H)
            bsnet_phase_sin = phase_sin[:, center, :]
            bsnet_phase_cos = phase_cos[:, center, :]

            # Broadcast U-Net phase from (B, T-1, H) to (B, T-1, H, W) and concat
            B, T_minus_1, H, W = unet_in.shape
            u_sin = unet_phase_sin.unsqueeze(-1).expand(B, T_minus_1, H, W)
            u_cos = unet_phase_cos.unsqueeze(-1).expand(B, T_minus_1, H, W)
            unet_in = torch.cat([unet_in, u_sin, u_cos], dim=1)

        if self.bp:
            unet_out = self.forward_unet(unet_in)
            x = unet_out
        elif self.twod:
            unet_out = None
            x = self.forward_bsnet(bsnet_in, unet_out, bsnet_phase_sin, bsnet_phase_cos)
        else:
            unet_out = self.forward_unet(unet_in)
            bsnet_out = self.forward_bsnet(
                bsnet_in, unet_out, bsnet_phase_sin, bsnet_phase_cos
            )

            x = torch.cat([unet_out, bsnet_out], dim=1)

        for i in range(len(self.last_layers) // 3):
            x = self.last_layers[3 * i + 2](
                self.last_layers[3 * i + 1](self.last_layers[3 * i](x))
            )
        x = self.last_layers[-1](x)
        return x


if __name__ == "__main__":

    def weights_init_normalized(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find("Conv2d") != -1:
            # torch.nn.init.ones_(m.weight)
            torch.nn.init.constant_(m.weight, 1 / (m.weight.numel() * 1))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif classname.find("ConvHole2D") != -1:
            # torch.nn.init.ones_(m.weight)
            torch.nn.init.constant_(
                m.weight,
                1 / ((m.weight.numel() - m.weight.size(0) * m.weight.size(1)) * 1),
            )
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    ch = 61

    model = SUPPORT(
        in_channels=ch,
        mid_channels=[16, 32, 64, 128, 256],
        depth=6,
        blind_conv_channels=4,
        one_by_one_channels=[32, 16],
        last_layer_channels=[4, 1],
        bs_size=[1, 1],
        bp=True,
    )
    model.apply(weights_init_normalized)

    import numpy as np

    inp = torch.zeros(1, 61, 2048, 2048)
    inp[:, 20, 1024, 1024] = 100
    out = model(inp)

    rf = np.ma.log(out.detach().numpy()[0, 0])
    rf = rf.filled(rf.min())

    import matplotlib.pyplot as plt

    plt.imshow(rf)
    plt.show()

    if False:
        import skimage.io as skio

        data = skio.imread("./test_pilhankim.tif")
        print(data.shape)

        data[0, 100, 101] = 1000000
        data[0, 100, 102] = 1000000
        data[0, 100, 103] = 1000000
        data[0, 100, 104] = 1000000  # this is horizontal dimension

        import matplotlib.pyplot as plt

        plt.imshow(data[0, :, :])
        plt.show()

        def weights_init_normalized(m):
            classname = m.__class__.__name__
            # print(classname)
            if classname.find("Conv2d") != -1:
                # torch.nn.init.ones_(m.weight)
                torch.nn.init.constant_(m.weight, 1 / (m.weight.numel() * 1))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif classname.find("ConvHole2D") != -1:
                # torch.nn.init.ones_(m.weight)
                torch.nn.init.constant_(
                    m.weight,
                    1 / ((m.weight.numel() - m.weight.size(0) * m.weight.size(1)) * 1),
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        ch = 1

        model = SUPPORT(
            in_channels=ch,
            mid_channels=[16, 32, 64, 128, 256],
            depth=5,
            blind_conv_channels=4,
            one_by_one_channels=[32, 16],
            last_layer_channels=[4, 1],
            bs_size=[1, 1],
            bp=False,
        )

        model.apply(weights_init_normalized)

        # print(model)

        import torch

        a = torch.zeros(1, ch, 128, 128)
        a[:, ch // 2, 64, 64] = 1000
        a[:, ch // 2, 64, 65] = 1000

        out = model(a)
        print(out[0, 0, 64, 64])
        print(out[0, 0, 64, 65])

        a[:, ch // 2, 64, 64] = 1000
        a[:, ch // 2, 64, 65] = 100000
        # a[:, ch // 2, 64, 65] = 2
        out = model(a)
        print(out[0, 0, 64, 64])
        print(out[0, 0, 64, 65])

        a[:, ch // 2, 63, 64] = 2
        a[:, ch // 2, 64, 64] = 2
        # a[:, ch // 2 + 1, 64, 65] = 2
        out = model(a)
        print(out[0, 0, 64, 64])
        print(out[0, 0, 64, 65])

        import matplotlib.pyplot as plt

        plt.imshow(out[0, 0, :, :].detach().numpy())
        plt.show()

        pass
