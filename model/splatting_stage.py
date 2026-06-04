"""
Learnable Splatting Stage for SUPPORT Model

This module implements a differentiable splatting operation that reconstructs
spatially-corrected frames from raw acquisition data with oscillating detector.

Input: [B, T, 2, H, W] where channel 0=eod (signal), channel 1=position (oscillation)
Output: [B, T, H', W'] spatially-corrected frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LearnableSplattingStage(nn.Module):
    """
    Differentiable splatting module with learnable reconstruction parameters.

    Parameters are initialized from manual splatting params and fine-tuned during training.
    """

    def __init__(self, initial_params, drop_lines=1):
        """
        Args:
            initial_params: Dict with splatting parameters from JSON
            drop_lines: Number of lines to drop from top (preprocessing)
        """
        super(LearnableSplattingStage, self).__init__()

        # Image dimensions will be inferred from input
        self.n_lines = None  # Set dynamically in forward()
        self.n_samples = None  # Set dynamically in forward()
        self.drop_lines = drop_lines

        # Extract parameters from initial_params dict
        splat_params = initial_params["splatting"]

        # Fixed parameters (architectural, not learned)
        self.eod_n_spots = int(splat_params["eod_n_spots"])
        self.upsample_factor = int(splat_params["splat_grid_upsample_factor"])
        self.eod_intensity_correction = splat_params.get(
            "eod_intensity_correction", True
        )

        # Output dimensions will be computed dynamically in forward()
        self.out_height = None
        self.out_width = None

        # Learnable parameters (8 total) - initialized from manual values
        self.eod_amplitude = nn.Parameter(
            torch.tensor(float(splat_params["eod_amplitude"]), dtype=torch.float32)
        )
        self.eod_phase_rad_even = nn.Parameter(
            torch.tensor(float(splat_params["eod_phase_rad"][0]), dtype=torch.float32)
        )
        self.eod_phase_rad_odd = nn.Parameter(
            torch.tensor(float(splat_params["eod_phase_rad"][1]), dtype=torch.float32)
        )
        self.reso_phase_shift_even = nn.Parameter(
            torch.tensor(
                float(splat_params["reso_phase_shift_px"][0]), dtype=torch.float32
            )
        )
        self.reso_phase_shift_odd = nn.Parameter(
            torch.tensor(
                float(splat_params["reso_phase_shift_px"][1]), dtype=torch.float32
            )
        )
        self.splat_sigma = nn.Parameter(
            torch.tensor(float(splat_params["splat_sigma_px"]), dtype=torch.float32)
        )
        self.reso_fill_fraction = nn.Parameter(
            torch.tensor(
                float(splat_params["reso_temporal_fill_fraction"]), dtype=torch.float32
            )
        )
        self.scan_aspect = nn.Parameter(
            torch.tensor(float(splat_params["scan_aspect"]), dtype=torch.float32)
        )

        # Store initial values for regularization loss
        initial_values = torch.tensor(
            [
                float(splat_params["eod_amplitude"]),
                float(splat_params["eod_phase_rad"][0]),
                float(splat_params["eod_phase_rad"][1]),
                float(splat_params["reso_phase_shift_px"][0]),
                float(splat_params["reso_phase_shift_px"][1]),
                float(splat_params["splat_sigma_px"]),
                float(splat_params["reso_temporal_fill_fraction"]),
                float(splat_params["scan_aspect"]),
            ],
            dtype=torch.float32,
        )
        self.register_buffer("initial_params_tensor", initial_values)

    def _osc_matched_filt_torch(self, x, w=None, phase_shift=0.0):
        """Torch port of eo_accelerated_2p.utils.osc_matched_filt for one line."""
        x = x.to(torch.float32)
        n = x.numel()
        if n == 0:
            return x, w

        n_idx = torch.arange(n, device=x.device, dtype=torch.float32)
        x0 = x - x.mean()
        win = torch.hann_window(n, periodic=False, device=x.device, dtype=torch.float32)

        if w is None:
            x_fft = torch.fft.rfft(x0 * win)
            mag = torch.abs(x_fft)
            if mag.numel() <= 1:
                return torch.zeros_like(x0), torch.tensor(0.0, device=x.device)

            k = int(torch.argmax(mag[1:]).item() + 1)
            delta = 0.0
            if 1 <= k < (mag.numel() - 1):
                a = mag[k - 1]
                b = mag[k]
                c = mag[k + 1]
                denom = a - 2.0 * b + c
                if torch.abs(denom) > 1e-24:
                    delta_t = 0.5 * (a - c) / denom
                    delta = float(torch.clamp(delta_t, -0.5, 0.5).item())
            w = torch.tensor(
                2.0 * math.pi * (k + delta) / n,
                device=x.device,
                dtype=torch.float32,
            )

        xw = torch.sum((x0 * win) * torch.exp(-1j * w * n_idx))
        win_sum = torch.sum(win)
        if win_sum <= 0:
            return torch.zeros_like(x0), w

        amp = (2.0 / win_sum) * torch.abs(xw)
        phi = torch.angle(xw)
        y = amp * torch.cos(w * n_idx + phi + phase_shift)
        return y, w

    def _build_scan_model(
        self,
        h,
        w_local,
        out_h,
        out_w,
        y_offset=0,
        x_offset=0,
        w_global=None,
    ):
        """Build EO-style scan model with even/odd masks and phase shifts."""
        fill = torch.clamp(self.reso_fill_fraction, 1e-3, 0.999)
        # For patch training/inference, keep a local dense raster by using local width
        # for modeled line length and treat x_offset as a phase shift term.
        n_modeled_line = int(w_local / float(fill.item()))
        n_blanked = n_modeled_line - w_local
        n_total = h * n_modeled_line

        t = torch.arange(n_total, device=self.eod_amplitude.device, dtype=torch.float32)
        y = torch.linspace(0, out_h, n_total, device=self.eod_amplitude.device)

        spatial_fill_fraction = torch.sin(fill * math.pi / 2)
        x = torch.cos(math.pi / n_modeled_line * t) / spatial_fill_fraction
        x = (x + 1.0) / 2.0 * out_w

        x_phase = float(x_offset) % float(n_modeled_line)
        odd_mask = (
            torch.arange(w_local, device=self.eod_amplitude.device, dtype=torch.float32)
            + x_phase
            + n_blanked / 2
            - self.reso_phase_shift_even
        ).to(torch.int64)
        even_mask = (
            torch.arange(w_local, device=self.eod_amplitude.device, dtype=torch.float32)
            + x_phase
            + n_blanked / 2
            + self.reso_phase_shift_odd
        ).to(torch.int64)

        odd_mask = torch.remainder(odd_mask, n_modeled_line)
        even_mask = torch.remainder(even_mask, n_modeled_line)

        scan_v = torch.zeros(
            h, w_local, device=self.eod_amplitude.device, dtype=torch.float32
        )
        scan_u = torch.zeros(
            h, w_local, device=self.eod_amplitude.device, dtype=torch.float32
        )
        for i in range(h):
            base = i * n_modeled_line
            global_i = int(y_offset) + i
            if global_i % 2 == 0:
                idx = even_mask + base
            else:
                idx = odd_mask + base
            scan_v[i, :] = y[idx]
            scan_u[i, :] = x[idx]

        return scan_v, scan_u

    def _bias_correction_torch(self, a, b, eps=1e-12):
        """Torch port of eo_accelerated_2p.utils.bias_correction.

        a, b: [T, H, W]
        Returns corrected a with same shape.
        """
        af = a.reshape(-1).to(torch.float32)
        bf = b.reshape(-1).to(torch.float32)

        mean_a = af.mean()
        std_a = af.std(unbiased=False)
        mean_b = bf.mean()

        mean_b2 = torch.dot(bf, bf) / bf.numel()
        var_b = mean_b2 - mean_b * mean_b
        if (not torch.isfinite(var_b)) or var_b <= eps:
            return a

        mean_ab = torch.dot(af, bf) / bf.numel()
        cov_ab = mean_ab - mean_a * mean_b
        alpha = cov_ab / var_b
        if not torch.isfinite(alpha):
            return a

        out = a.to(torch.float32).clone()
        outf = out.reshape(-1)
        outf -= alpha * bf
        outf += alpha * mean_b

        if std_a > eps:
            std_out = outf.std(unbiased=False)
            if torch.isfinite(std_out) and std_out > eps:
                outf -= mean_a
                outf *= std_a / std_out
                outf += mean_a

        return out

    def _compute_frame_base_coords(
        self, eod_frame, position_frame, scan_v, scan_u, out_h
    ):
        """Compute per-point base coordinates before learned offsets.

        Args:
            eod_frame: [1, H, W]
            position_frame: [1, H, W]
            scan_v/scan_u: [H, W]
            out_h: output height in splat space

        Returns:
            v, u, s each [1, H, W]
        """
        _, h, w = eod_frame.shape
        v = scan_v.clone()
        u = scan_u.clone()
        s = torch.empty_like(v)
        eod_freq = None

        for i in range(h):
            line_s = eod_frame[0, i, :]
            line_eod = position_frame[0, i, :]
            phase_shift = (
                self.eod_phase_rad_even if (i % 2 == 0) else self.eod_phase_rad_odd
            )
            line_eod, eod_freq = self._osc_matched_filt_torch(
                line_eod, eod_freq, phase_shift=phase_shift
            )

            line_min = torch.min(line_eod)
            line_max = torch.max(line_eod)
            line_eod = (line_eod - line_min) / (line_max - line_min + 1e-8) - 0.5
            line_eod = self.eod_amplitude * out_h * line_eod

            if i % 2 == 0:
                v[i, :] = v[i, :] + line_eod
                s[i, :] = line_s
            else:
                v[i, :] = v[i, :] + torch.flip(line_eod, dims=[0])
                s[i, :] = torch.flip(line_s, dims=[0])

        return v.unsqueeze(0), u.unsqueeze(0), s.unsqueeze(0)

    def _prepare_sample(self, x_b, y_off, x_off, full_n_samples_b):
        """Prepare per-sample tensors and scan model with origin handling."""
        effective_drop = self.drop_lines

        if effective_drop > 0:
            x_b = x_b[:, :, :, effective_drop:, :]

        h_local = x_b.shape[3]
        w_local = x_b.shape[4]
        y_global = y_off + effective_drop

        if full_n_samples_b is None:
            w_global = w_local
        else:
            w_global = int(full_n_samples_b)

        self.n_lines = h_local
        self.n_samples = w_local
        self.out_height = int(h_local * self.eod_n_spots * self.upsample_factor)
        self.out_width = int(self.out_height * float(self.scan_aspect.item()))

        eod_signal = x_b[:, :, 0, :, :]
        position_signal = x_b[:, :, 1, :, :]

        if self.eod_intensity_correction:
            position_signal[0] = self._bias_correction_torch(
                position_signal[0], eod_signal[0], eps=1e-2
            )

        scan_v, scan_u = self._build_scan_model(
            h_local,
            w_local,
            self.out_height,
            self.out_width,
            y_offset=y_global,
            x_offset=x_off,
            w_global=w_global,
        )
        return eod_signal, position_signal, scan_v, scan_u

    def compute_base_coordinates(self, x, patch_origin=None, full_n_samples=None):
        """Compute base splat coordinates (u0, v0) for each raw point.

        Returns:
            value: [B, T, H', W']? no, returns local raw-grid tensors:
            value [B,T,Hd,W], u0 [B,T,Hd,W], v0 [B,T,Hd,W]
            where Hd = H - drop_lines (fixed drop behavior)
        """
        b, t, c, _, _ = x.shape
        assert c == 2

        values = []
        u0_all = []
        v0_all = []

        for bi in range(b):
            if patch_origin is not None:
                y_off = int(patch_origin[bi, 0].item())
                x_off = int(patch_origin[bi, 1].item())
            else:
                y_off = 0
                x_off = 0

            full_w = None if full_n_samples is None else int(full_n_samples[bi].item())
            x_b = x[bi : bi + 1]
            eod_signal, position_signal, scan_v, scan_u = self._prepare_sample(
                x_b, y_off, x_off, full_w
            )

            v_seq = []
            u_seq = []
            val_seq = []
            for ti in range(t):
                v_t, u_t, s_t = self._compute_frame_base_coords(
                    eod_signal[:, ti],
                    position_signal[:, ti],
                    scan_v,
                    scan_u,
                    self.out_height,
                )
                v_seq.append(v_t[0])
                u_seq.append(u_t[0])
                val_seq.append(s_t[0])

            v0_all.append(torch.stack(v_seq, dim=0))
            u0_all.append(torch.stack(u_seq, dim=0))
            values.append(torch.stack(val_seq, dim=0))

        return (
            torch.stack(values, dim=0),
            torch.stack(u0_all, dim=0),
            torch.stack(v0_all, dim=0),
        )

    def _extract_oscillation_phase(self, position_line, is_odd):
        """
        Extract oscillation phase from position signal for a single line.
        Simplified version without matched filtering for speed.

        Args:
            position_line: [B, W] position signal for one line
            is_odd: Boolean, whether this is an odd line

        Returns:
            normalized oscillation signal [B, W]
        """
        # Normalize to [-0.5, 0.5] range
        pos_min = position_line.min(dim=-1, keepdim=True)[0]
        pos_max = position_line.max(dim=-1, keepdim=True)[0]
        pos_norm = (position_line - pos_min) / (pos_max - pos_min + 1e-8) - 0.5

        # Apply phase shift
        if is_odd:
            phase_shift = self.eod_phase_rad_odd
        else:
            phase_shift = self.eod_phase_rad_even

        # Convert phase shift to sample shift (simplified)
        # In the original code, this involves rolling by pixel amount
        # Here we approximate with a slight DC shift
        pos_norm = pos_norm + phase_shift * 0.1  # Small phase-dependent offset

        return pos_norm

    def forward(self, x, patch_origin=None, full_n_samples=None, learned_offsets=None):
        """
        Forward pass: splat all frames in the batch.

        Args:
            x: [B, T, 2, H, W] where dim 2 is [eod, position]

        Returns:
            [B, T, H', W'] splatted frames
        """
        B, T, C, H, W = x.shape
        assert C == 2, f"Expected 2 channels (eod, position), got {C}"

        # Infer dimensions from input (H = n_lines, W = n_samples)
        input_n_lines = H
        input_n_samples = W

        # Process each sample independently so patch origin offsets can be honored
        splatted_batch = []
        for b in range(B):
            if patch_origin is not None:
                y_off = int(patch_origin[b, 0].item())
                x_off = int(patch_origin[b, 1].item())
            else:
                y_off = 0
                x_off = 0

            x_b = x[b : b + 1]
            full_w = None if full_n_samples is None else int(full_n_samples[b].item())
            eod_signal, position_signal, scan_v, scan_u = self._prepare_sample(
                x_b, y_off, x_off, full_w
            )

            sample_frames = []
            for t in range(T):
                offsets_t = None
                if learned_offsets is not None:
                    offsets_t = learned_offsets[b : b + 1, t]
                    if (
                        offsets_t.shape[-2] != eod_signal.shape[-2]
                        and self.drop_lines > 0
                    ):
                        offsets_t = offsets_t[:, :, self.drop_lines :, :]

                splatted_t = self._splat_single_frame(
                    eod_signal[:, t],
                    position_signal[:, t],
                    scan_v,
                    scan_u,
                    offsets_frame=offsets_t,
                )
                sample_frames.append(splatted_t[0])

            splatted_batch.append(torch.stack(sample_frames, dim=0))

        return torch.stack(splatted_batch, dim=0)

    def _splat_single_frame(self, eod, position, scan_v, scan_u, offsets_frame=None):
        """
        Splat a single frame (batched across B).

        Args:
            eod: [B, H, W] signal values
            position: [B, H, W] oscillation signal

        Returns:
            [B, H', W'] splatted frame
        """
        B, H, W = eod.shape
        device = eod.device

        # Output dimensions
        out_h = int(self.n_lines * self.eod_n_spots * self.upsample_factor)
        out_w = int(out_h * self.scan_aspect)

        final_h = out_h // self.upsample_factor if self.upsample_factor > 1 else out_h
        final_w = out_w // self.upsample_factor if self.upsample_factor > 1 else out_w
        output = torch.zeros(B, final_h, final_w, device=device, dtype=torch.float32)
        sigma = self.splat_sigma * self.upsample_factor

        for b in range(B):
            v = scan_v.clone()
            u = scan_u

            if offsets_frame is not None:
                # offsets_frame: [1, 2, H, W] (dx, dy)
                dx = offsets_frame[b, 0]
                dy = offsets_frame[b, 1] * out_h
                v = v + dy
                u = u + dx

            s = torch.empty_like(v)
            eod_freq = None

            for i in range(H):
                line_s = eod[b, i, :]
                line_eod = position[b, i, :]
                phase_shift = (
                    self.eod_phase_rad_even if (i % 2 == 0) else self.eod_phase_rad_odd
                )
                line_eod, eod_freq = self._osc_matched_filt_torch(
                    line_eod, eod_freq, phase_shift=phase_shift
                )

                line_min = torch.min(line_eod)
                line_max = torch.max(line_eod)
                line_eod = (line_eod - line_min) / (line_max - line_min + 1e-8) - 0.5
                line_eod = self.eod_amplitude * out_h * line_eod

                if i % 2 == 0:
                    v[i, :] = v[i, :] + line_eod
                    s[i, :] = line_s
                else:
                    v[i, :] = v[i, :] + torch.flip(line_eod, dims=[0])
                    s[i, :] = torch.flip(line_s, dims=[0])

            ii = torch.clamp(v, 0, out_h - 1).to(torch.int64)
            jj = torch.clamp(u, 0, out_w - 1).to(torch.int64)
            idx = (ii.reshape(-1) * out_w + jj.reshape(-1)).to(torch.int64)
            weights = s.reshape(-1).to(torch.float32)

            numerator = torch.bincount(idx, weights=weights, minlength=out_h * out_w)
            numerator = numerator.reshape(out_h, out_w)
            denominator = torch.bincount(idx, minlength=out_h * out_w).to(torch.float32)
            denominator = denominator.reshape(out_h, out_w)

            numerator = self._gaussian_blur(numerator.unsqueeze(0), sigma).squeeze(0)
            denominator = self._gaussian_blur(denominator.unsqueeze(0), sigma).squeeze(
                0
            )

            dst = numerator / torch.clamp(denominator, min=1e-18)

            if self.upsample_factor > 1:
                dst = (
                    F.interpolate(
                        dst.unsqueeze(0).unsqueeze(0),
                        size=(final_h, final_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            output[b] = dst

        return output

    def _gaussian_blur(self, x, sigma):
        """
        Apply Gaussian blur using depthwise conv2d.

        Args:
            x: [B, H, W] tensor
            sigma: Blur sigma

        Returns:
            [B, H, W] blurred tensor
        """
        # Clamp sigma to reasonable range
        sigma = torch.clamp(sigma, 0.1, 10.0)

        # Compute kernel size
        kernel_size = int(6 * sigma.item() + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x_coord = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        center = kernel_size // 2

        gaussian = torch.exp(
            -((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma**2)
        )
        gaussian = gaussian / gaussian.sum()

        # Reshape for conv2d: [out_channels, in_channels/groups, kH, kW]
        # For depthwise: groups=in_channels, so in_channels/groups=1
        kernel = gaussian.view(1, 1, kernel_size, kernel_size)

        # Add channel dimension
        x = x.unsqueeze(1)  # [B, 1, H, W]

        # Apply convolution with reflect padding (matches scipy.ndimage.gaussian_filter)
        padding = kernel_size // 2
        x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        blurred = F.conv2d(x, kernel, padding=0, groups=1)

        return blurred.squeeze(1)  # [B, H, W]

    def get_params_dict(self):
        """Return current parameter values as dict (for logging)"""
        return {
            "eod_amplitude": self.eod_amplitude.item(),
            "eod_phase_rad_even": self.eod_phase_rad_even.item(),
            "eod_phase_rad_odd": self.eod_phase_rad_odd.item(),
            "reso_phase_shift_even": self.reso_phase_shift_even.item(),
            "reso_phase_shift_odd": self.reso_phase_shift_odd.item(),
            "splat_sigma": self.splat_sigma.item(),
            "reso_fill_fraction": self.reso_fill_fraction.item(),
            "scan_aspect": self.scan_aspect.item(),
        }

    def get_params_tensor(self):
        """Return current parameters as tensor (for regularization)"""
        return torch.stack(
            [
                self.eod_amplitude,
                self.eod_phase_rad_even,
                self.eod_phase_rad_odd,
                self.reso_phase_shift_even,
                self.reso_phase_shift_odd,
                self.splat_sigma,
                self.reso_fill_fraction,
                self.scan_aspect,
            ]
        )
