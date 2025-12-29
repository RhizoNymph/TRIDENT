"""
VRAM-aware scheduling for GPU-based slide encoding.

This module provides:
- EncoderMemoryProfile: Memory characteristics for each encoder model
- VRAMEstimator: Estimates optimal batch sizes based on available VRAM
- GPUScheduler: Distributes WSIs across multiple GPUs based on VRAM capacity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import h5py


@dataclass
class EncoderMemoryProfile:
    """Memory profile for a patch encoder model."""
    enc_name: str
    model_size_mb: float          # Approximate model weights in MB
    precision: torch.dtype        # float16, bfloat16, or float32
    input_size: int               # Input image size (224, 256, 384, 448, 518)
    output_dim: int               # Feature dimension output
    activation_multiplier: float  # Empirical factor for intermediate activations (2.0-4.0)

    @property
    def bytes_per_element(self) -> int:
        """Number of bytes per tensor element based on precision."""
        if self.precision in (torch.float16, torch.bfloat16):
            return 2
        return 4


# Memory profiles for all supported encoders
# Model sizes are approximate and based on parameter counts
ENCODER_PROFILES: Dict[str, EncoderMemoryProfile] = {
    # CNN-based (smaller, faster)
    "resnet50": EncoderMemoryProfile(
        enc_name="resnet50", model_size_mb=100, precision=torch.float32,
        input_size=224, output_dim=2048, activation_multiplier=2.0
    ),
    "ctranspath": EncoderMemoryProfile(
        enc_name="ctranspath", model_size_mb=110, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=2.5
    ),

    # ViT-Small models (~22M params)
    "kaiko-vits8": EncoderMemoryProfile(
        enc_name="kaiko-vits8", model_size_mb=85, precision=torch.float32,
        input_size=224, output_dim=384, activation_multiplier=2.5
    ),
    "kaiko-vits16": EncoderMemoryProfile(
        enc_name="kaiko-vits16", model_size_mb=85, precision=torch.float32,
        input_size=224, output_dim=384, activation_multiplier=2.5
    ),
    "lunit-vits8": EncoderMemoryProfile(
        enc_name="lunit-vits8", model_size_mb=85, precision=torch.float32,
        input_size=224, output_dim=384, activation_multiplier=2.5
    ),

    # ViT-Base models (~86M params)
    "phikon": EncoderMemoryProfile(
        enc_name="phikon", model_size_mb=330, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),
    "phikon_v2": EncoderMemoryProfile(
        enc_name="phikon_v2", model_size_mb=330, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),
    "hibou_l": EncoderMemoryProfile(
        enc_name="hibou_l", model_size_mb=330, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),
    "kaiko-vitb8": EncoderMemoryProfile(
        enc_name="kaiko-vitb8", model_size_mb=170, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),
    "kaiko-vitb16": EncoderMemoryProfile(
        enc_name="kaiko-vitb16", model_size_mb=170, precision=torch.float32,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),
    "conch_v1": EncoderMemoryProfile(
        enc_name="conch_v1", model_size_mb=350, precision=torch.float32,
        input_size=224, output_dim=512, activation_multiplier=3.0
    ),
    "midnight12k": EncoderMemoryProfile(
        enc_name="midnight12k", model_size_mb=330, precision=torch.float16,
        input_size=224, output_dim=768, activation_multiplier=3.0
    ),

    # ViT-Large models (~300M params)
    "uni_v1": EncoderMemoryProfile(
        enc_name="uni_v1", model_size_mb=1200, precision=torch.float16,
        input_size=224, output_dim=1024, activation_multiplier=3.5
    ),
    "kaiko-vitl14": EncoderMemoryProfile(
        enc_name="kaiko-vitl14", model_size_mb=1200, precision=torch.float32,
        input_size=518, output_dim=1024, activation_multiplier=3.5
    ),
    "conch_v15": EncoderMemoryProfile(
        enc_name="conch_v15", model_size_mb=600, precision=torch.float16,
        input_size=448, output_dim=512, activation_multiplier=3.0
    ),
    "musk": EncoderMemoryProfile(
        enc_name="musk", model_size_mb=1200, precision=torch.float16,
        input_size=384, output_dim=768, activation_multiplier=3.0
    ),

    # ViT-Huge models (~600M+ params)
    "virchow": EncoderMemoryProfile(
        enc_name="virchow", model_size_mb=2500, precision=torch.float16,
        input_size=224, output_dim=2560, activation_multiplier=3.5
    ),
    "virchow2": EncoderMemoryProfile(
        enc_name="virchow2", model_size_mb=2500, precision=torch.float16,
        input_size=224, output_dim=2560, activation_multiplier=3.5
    ),

    # ViT-Giant models (~1B+ params)
    "uni_v2": EncoderMemoryProfile(
        enc_name="uni_v2", model_size_mb=4000, precision=torch.bfloat16,
        input_size=224, output_dim=1536, activation_multiplier=3.5
    ),
    "gigapath": EncoderMemoryProfile(
        enc_name="gigapath", model_size_mb=4500, precision=torch.float16,
        input_size=224, output_dim=1536, activation_multiplier=3.5
    ),
    "hoptimus0": EncoderMemoryProfile(
        enc_name="hoptimus0", model_size_mb=4000, precision=torch.float16,
        input_size=224, output_dim=1536, activation_multiplier=3.5
    ),
    "hoptimus1": EncoderMemoryProfile(
        enc_name="hoptimus1", model_size_mb=4000, precision=torch.float16,
        input_size=224, output_dim=1536, activation_multiplier=3.5
    ),
    "open-midnight": EncoderMemoryProfile(
        enc_name="open-midnight", model_size_mb=4000, precision=torch.float16,
        input_size=224, output_dim=1536, activation_multiplier=3.5
    ),
}


class VRAMEstimator:
    """
    Estimates VRAM requirements and optimal batch sizes for patch feature extraction.

    Uses pre-flight estimation based on encoder memory profiles and available GPU memory.
    """

    SAFETY_MARGIN = 0.85  # Use only 85% of available VRAM
    CUDA_OVERHEAD_MB = 500  # Base CUDA overhead in MB
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 2048

    def __init__(
        self,
        encoder: torch.nn.Module,
        device: str = 'cuda:0',
        safety_margin: float = 0.85
    ):
        """
        Initialize VRAM estimator for a given encoder.

        Parameters
        ----------
        encoder : torch.nn.Module
            The patch encoder model (must have enc_name and precision attributes)
        device : str
            CUDA device string (e.g., 'cuda:0')
        safety_margin : float
            Fraction of available VRAM to use (0.0-1.0). Default: 0.85
        """
        self.encoder = encoder
        self.device = device
        self.safety_margin = safety_margin
        self.device_idx = self._parse_device_index(device)
        self.profile = self._get_encoder_profile()

    def _parse_device_index(self, device: str) -> int:
        """Extract GPU index from device string."""
        if ':' in device:
            return int(device.split(':')[1])
        return 0

    def _get_encoder_profile(self) -> EncoderMemoryProfile:
        """
        Get memory profile for the encoder.

        Falls back to empirical measurement if encoder not in registry.
        """
        enc_name = getattr(self.encoder, 'enc_name', None)

        if enc_name and enc_name in ENCODER_PROFILES:
            return ENCODER_PROFILES[enc_name]

        # Fallback: create profile from encoder attributes
        precision = getattr(self.encoder, 'precision', torch.float32)

        # Estimate model size from parameters
        param_count = sum(p.numel() for p in self.encoder.parameters())
        bytes_per_param = 2 if precision in (torch.float16, torch.bfloat16) else 4
        model_size_mb = (param_count * bytes_per_param) / (1024 ** 2)

        # Try to get output dim by running a dummy forward pass
        output_dim = 768  # Default
        try:
            self.encoder.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                if precision != torch.float32:
                    with torch.autocast(device_type='cuda', dtype=precision):
                        out = self.encoder(dummy)
                else:
                    out = self.encoder(dummy)
                output_dim = out.shape[-1]
        except Exception:
            pass

        return EncoderMemoryProfile(
            enc_name=enc_name or "unknown",
            model_size_mb=model_size_mb,
            precision=precision,
            input_size=224,
            output_dim=output_dim,
            activation_multiplier=3.0
        )

    def get_total_vram(self) -> float:
        """Get total VRAM on target device in MB."""
        props = torch.cuda.get_device_properties(self.device_idx)
        return props.total_memory / (1024 ** 2)

    def get_available_vram(self) -> float:
        """Get available (free) VRAM on target device in MB."""
        total = self.get_total_vram()
        allocated = torch.cuda.memory_allocated(self.device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device_idx) / (1024 ** 2)
        # Use allocated as the main metric, but also consider reserved
        return total - max(allocated, reserved)

    def estimate_batch_memory_mb(self, batch_size: int, patch_size: Optional[int] = None) -> float:
        """
        Estimate total VRAM needed for a batch of patches.

        Parameters
        ----------
        batch_size : int
            Number of patches per batch
        patch_size : int, optional
            Patch size in pixels. If None, uses profile's input_size.

        Returns
        -------
        float
            Estimated memory usage in MB
        """
        if patch_size is None:
            patch_size = self.profile.input_size

        bpe = self.profile.bytes_per_element

        # Input tensor: batch_size * 3 * patch_size^2 * bytes_per_element
        input_size_mb = (batch_size * 3 * patch_size * patch_size * bpe) / (1024 ** 2)

        # Intermediate activations (empirical multiplier for ViT architectures)
        activations_mb = input_size_mb * self.profile.activation_multiplier

        # Output embeddings: batch_size * output_dim * bytes_per_element
        output_mb = (batch_size * self.profile.output_dim * bpe) / (1024 ** 2)

        return input_size_mb + activations_mb + output_mb

    def get_optimal_batch_size(self, patch_size: Optional[int] = None) -> int:
        """
        Calculate optimal batch size given available VRAM.

        Uses binary search to find the largest batch size that fits in available memory.

        Parameters
        ----------
        patch_size : int, optional
            Patch size in pixels. If None, uses profile's input_size.

        Returns
        -------
        int
            Optimal batch size
        """
        available = self.get_available_vram()
        usable = (available * self.safety_margin) - self.CUDA_OVERHEAD_MB

        # Subtract model size (already on GPU or will be loaded)
        usable -= self.profile.model_size_mb

        if usable <= 0:
            return self.MIN_BATCH_SIZE

        # Binary search for optimal batch size
        low, high = self.MIN_BATCH_SIZE, self.MAX_BATCH_SIZE
        result = self.MIN_BATCH_SIZE

        while low <= high:
            mid = (low + high) // 2
            mem_needed = self.estimate_batch_memory_mb(mid, patch_size)

            if mem_needed <= usable:
                result = mid
                low = mid + 1
            else:
                high = mid - 1

        return result

    def estimate_wsi_memory_requirement(self, coords_path: str) -> Dict[str, float]:
        """
        Pre-flight estimation of WSI processing requirements.

        Parameters
        ----------
        coords_path : str
            Path to the HDF5 file containing patch coordinates

        Returns
        -------
        dict
            Dictionary with:
            - 'num_patches': Total patches in WSI
            - 'peak_memory_mb': Estimated peak VRAM usage
            - 'recommended_batch_size': Suggested batch size
        """
        with h5py.File(coords_path, 'r') as f:
            num_patches = f['coords'].shape[0]
            patch_size = dict(f['coords'].attrs).get('patch_size', 224)

        optimal_batch = self.get_optimal_batch_size(patch_size)
        peak_memory = self.estimate_batch_memory_mb(optimal_batch, patch_size)
        peak_memory += self.profile.model_size_mb

        return {
            'num_patches': num_patches,
            'peak_memory_mb': peak_memory,
            'recommended_batch_size': optimal_batch,
        }


class GPUScheduler:
    """
    Schedules WSI processing across multiple GPUs based on VRAM availability.

    Supports multiple scheduling strategies:
    - load_balance: Balance by estimated processing load (patch count)
    - round_robin: Simple rotation across GPUs
    - memory_aware: Consider per-WSI memory requirements
    """

    def __init__(
        self,
        gpus: List[int],
        encoder: torch.nn.Module,
        strategy: str = 'load_balance'
    ):
        """
        Initialize GPU scheduler.

        Parameters
        ----------
        gpus : List[int]
            List of GPU indices to use
        encoder : torch.nn.Module
            The patch encoder (used for memory profiling)
        strategy : str
            Scheduling strategy: 'load_balance', 'round_robin', or 'memory_aware'
        """
        self.gpus = gpus
        self.encoder = encoder
        self.strategy = strategy

        # Create estimators for each GPU
        self.estimators = {
            gpu: VRAMEstimator(encoder, f'cuda:{gpu}')
            for gpu in gpus
        }

        # Track current load on each GPU (in patches)
        self.gpu_loads: Dict[int, int] = {gpu: 0 for gpu in gpus}

        # Round-robin counter
        self._rr_counter = 0

    def get_gpu_vram_info(self) -> Dict[int, Dict[str, float]]:
        """Get VRAM information for all GPUs."""
        info = {}
        for gpu, estimator in self.estimators.items():
            info[gpu] = {
                'total_mb': estimator.get_total_vram(),
                'available_mb': estimator.get_available_vram(),
                'optimal_batch_size': estimator.get_optimal_batch_size(),
            }
        return info

    def estimate_wsi_load(self, coords_path: str) -> int:
        """
        Estimate processing load for a WSI (number of patches).

        Parameters
        ----------
        coords_path : str
            Path to coordinates HDF5 file

        Returns
        -------
        int
            Number of patches (proxy for processing load)
        """
        try:
            with h5py.File(coords_path, 'r') as f:
                return f['coords'].shape[0]
        except Exception:
            return 1000  # Default estimate

    def assign_wsi_to_gpu(self, wsi_name: str, coords_path: str) -> Tuple[int, int]:
        """
        Assign a WSI to the most suitable GPU.

        Parameters
        ----------
        wsi_name : str
            Name of the WSI
        coords_path : str
            Path to coordinates file

        Returns
        -------
        Tuple[int, int]
            (gpu_index, recommended_batch_size)
        """
        if self.strategy == 'round_robin':
            gpu = self.gpus[self._rr_counter % len(self.gpus)]
            self._rr_counter += 1
            batch_size = self.estimators[gpu].get_optimal_batch_size()
            return gpu, batch_size

        elif self.strategy == 'memory_aware':
            # Assign to GPU with most available VRAM
            best_gpu = max(
                self.gpus,
                key=lambda g: self.estimators[g].get_available_vram()
            )
            batch_size = self.estimators[best_gpu].get_optimal_batch_size()
            return best_gpu, batch_size

        else:  # load_balance (default)
            # Assign to GPU with lowest current load
            best_gpu = min(self.gpus, key=lambda g: self.gpu_loads[g])

            # Update load estimate
            load = self.estimate_wsi_load(coords_path)
            self.gpu_loads[best_gpu] += load

            batch_size = self.estimators[best_gpu].get_optimal_batch_size()
            return best_gpu, batch_size

    def release_gpu(self, gpu: int, num_patches: int) -> None:
        """
        Release load from a GPU after WSI processing completes.

        Parameters
        ----------
        gpu : int
            GPU index
        num_patches : int
            Number of patches that were processed
        """
        self.gpu_loads[gpu] = max(0, self.gpu_loads[gpu] - num_patches)

    def get_batch_sizes(self) -> Dict[int, int]:
        """Get optimal batch size for each GPU."""
        return {
            gpu: estimator.get_optimal_batch_size()
            for gpu, estimator in self.estimators.items()
        }

    def assign_wsis_to_gpus(
        self,
        wsi_coords_list: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, int, int]]:
        """
        Assign multiple WSIs to GPUs.

        Parameters
        ----------
        wsi_coords_list : List[Tuple[str, str]]
            List of (wsi_name, coords_path) tuples

        Returns
        -------
        List[Tuple[str, str, int, int]]
            List of (wsi_name, coords_path, gpu_index, batch_size) tuples
        """
        assignments = []

        if self.strategy == 'load_balance':
            # Sort WSIs by patch count (largest first) for better load balancing
            wsi_loads = []
            for wsi_name, coords_path in wsi_coords_list:
                load = self.estimate_wsi_load(coords_path)
                wsi_loads.append((wsi_name, coords_path, load))

            wsi_loads.sort(key=lambda x: x[2], reverse=True)

            for wsi_name, coords_path, _ in wsi_loads:
                gpu, batch_size = self.assign_wsi_to_gpu(wsi_name, coords_path)
                assignments.append((wsi_name, coords_path, gpu, batch_size))
        else:
            for wsi_name, coords_path in wsi_coords_list:
                gpu, batch_size = self.assign_wsi_to_gpu(wsi_name, coords_path)
                assignments.append((wsi_name, coords_path, gpu, batch_size))

        return assignments


def get_vram_estimator(
    encoder: torch.nn.Module,
    device: str = 'cuda:0',
    safety_margin: float = 0.85
) -> VRAMEstimator:
    """
    Factory function to create a VRAM estimator.

    Parameters
    ----------
    encoder : torch.nn.Module
        The patch encoder model
    device : str
        CUDA device string
    safety_margin : float
        Fraction of available VRAM to use

    Returns
    -------
    VRAMEstimator
        Configured VRAM estimator
    """
    return VRAMEstimator(encoder, device, safety_margin)


def print_gpu_memory_status(gpus: Optional[List[int]] = None) -> None:
    """
    Print memory status for specified GPUs (useful for debugging).

    Parameters
    ----------
    gpus : List[int], optional
        List of GPU indices. If None, uses all available GPUs.
    """
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))

    print("\n=== GPU Memory Status ===")
    for gpu in gpus:
        props = torch.cuda.get_device_properties(gpu)
        total = props.total_memory / (1024 ** 3)  # GB
        allocated = torch.cuda.memory_allocated(gpu) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(gpu) / (1024 ** 3)
        free = total - reserved

        print(f"GPU {gpu} ({props.name}):")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Free:      {free:.2f} GB")
    print("========================\n")
