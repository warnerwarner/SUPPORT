import argparse
import json
import os

import numpy as np
import torch
import zarr
import skimage.io as skio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.SUPPORT import SUPPORT


class TemporalRawWindowDataset(Dataset):
    def __init__(self, eod_arr, pos_arr, input_frames, eod_mean, eod_std):
        self.eod_arr = eod_arr
        self.pos_arr = pos_arr
        self.input_frames = input_frames
        self.half = input_frames // 2
        self.eod_mean = float(eod_mean)
        self.eod_std = float(eod_std)

        if eod_arr.shape != pos_arr.shape:
            raise ValueError(
                f"eod and position shape mismatch: {eod_arr.shape} vs {pos_arr.shape}"
            )
        if eod_arr.ndim != 3:
            raise ValueError(f"Expected [T,H,W], got {eod_arr.shape}")
        if eod_arr.shape[0] < input_frames:
            raise ValueError(
                f"Not enough frames ({eod_arr.shape[0]}) for input_frames={input_frames}"
            )

    def __len__(self):
        return self.eod_arr.shape[0] - self.input_frames + 1

    def __getitem__(self, idx):
        eod = self.eod_arr[idx : idx + self.input_frames].astype(np.float32)
        pos = self.pos_arr[idx : idx + self.input_frames].astype(np.float32)

        eod = (eod - self.eod_mean) / (self.eod_std + 1e-8)
        x = np.stack([eod, pos], axis=1)  # [T, 2, H, W]

        center_frame_idx = idx + self.half
        return torch.from_numpy(x), center_frame_idx


def build_model(args, device):
    with open(args.splatting_params_json, "r") as f:
        splatting_params = json.load(f)

    drop_lines = int(splatting_params.get("preprocessing", {}).get("drop_lines", 1))
    splatting_config = {
        "initial_params": splatting_params,
        "drop_lines": drop_lines,
    }

    model = SUPPORT(
        in_channels=args.input_frames,
        mid_channels=args.unet_channels,
        depth=args.depth,
        blind_conv_channels=args.blind_conv_channels,
        one_by_one_channels=args.one_by_one_channels,
        last_layer_channels=args.last_layer_channels,
        bs_size=args.bs_size,
        bp=args.bp,
        use_splatting=True,
        splatting_config=splatting_config,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def validate_splatting_file(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    with zarr.open(args.input_zarr, mode="r") as store:
        eod_arr = store["eod"]
        pos_arr = store["position"]

        eod_mean = eod_arr.attrs.get("mean", None)
        eod_std = eod_arr.attrs.get("std", None)
        if eod_mean is None or eod_std is None:
            sample = eod_arr[:]
            eod_mean = float(sample.mean())
            eod_std = float(sample.std())

        dataset = TemporalRawWindowDataset(
            eod_arr=eod_arr,
            pos_arr=pos_arr,
            input_frames=args.input_frames,
            eod_mean=eod_mean,
            eod_std=eod_std,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args, device)

    outputs = []
    center_indices = []
    with torch.no_grad():
        for x, center_idx in tqdm(loader, desc="validate_splatting"):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=args.use_amp and device.type == "cuda"
            ):
                y = model(x)  # [B, 1, H', W']
            outputs.append(y[:, 0].cpu().numpy())
            center_indices.append(center_idx.numpy())

    denoised = np.concatenate(outputs, axis=0)  # [N_windows, H', W']
    centers = np.concatenate(center_indices, axis=0)

    order = np.argsort(centers)
    denoised = denoised[order]
    centers = centers[order]

    if args.output_tif:
        os.makedirs(os.path.dirname(args.output_tif) or ".", exist_ok=True)
        skio.imsave(
            args.output_tif, denoised.astype(np.float32), metadata={"axes": "TYX"}
        )
    if args.output_npy:
        os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)
        np.save(args.output_npy, denoised.astype(np.float32))

    print(f"Saved {denoised.shape[0]} denoised center frames")
    print(f"Output shape: {denoised.shape} [T_valid, H', W']")
    print(f"Center frame range in original stack: {centers[0]}..{centers[-1]}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate SUPPORT model with use_splatting=True"
    )
    p.add_argument(
        "--input_zarr",
        type=str,
        required=True,
        help="Input zarr path with eod/position",
    )
    p.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model_*.pth"
    )
    p.add_argument(
        "--splatting_params_json",
        type=str,
        required=True,
        help="Path to splatting params JSON",
    )

    p.add_argument(
        "--output_tif", type=str, default=None, help="Optional output tif path"
    )
    p.add_argument(
        "--output_npy", type=str, default=None, help="Optional output npy path"
    )

    p.add_argument("--input_frames", type=int, default=61)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_cpu", action="store_true")

    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--blind_conv_channels", type=int, default=64)
    p.add_argument("--one_by_one_channels", type=int, nargs="+", default=[32, 16])
    p.add_argument("--last_layer_channels", type=int, nargs="+", default=[64, 32, 16])
    p.add_argument(
        "--unet_channels", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    p.add_argument("--bs_size", type=int, nargs="+", default=[1, 3])
    p.add_argument("--bp", action="store_true")

    args = p.parse_args()
    if args.output_tif is None and args.output_npy is None:
        raise ValueError("Provide at least one of --output_tif or --output_npy")
    if args.input_frames % 2 == 0:
        raise ValueError("--input_frames must be odd")
    return args


if __name__ == "__main__":
    validate_splatting_file(parse_args())
