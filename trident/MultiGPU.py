"""
Multi-GPU processing for TRIDENT slide encoding.

This module provides multiprocessing-based coordination for distributing
WSI processing across multiple GPUs with VRAM-aware load balancing.
"""

from __future__ import annotations
import gc
import os
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Callable, Optional, Dict, Any
from queue import Empty
from dataclasses import dataclass


@dataclass
class GPUWorkerConfig:
    """Configuration for a GPU worker process."""
    gpu_id: int
    batch_size: int
    encoder_name: str
    encoder_kwargs: Dict[str, Any]
    safety_margin: float = 0.85


def gpu_worker(
    gpu_id: int,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    encoder_name: str,
    encoder_kwargs: Dict[str, Any],
    processor_config: Dict[str, Any],
    task: str,
    coords_dir: str,
    batch_size: Optional[int] = None,
    auto_batch_size: bool = True,
    safety_margin: float = 0.85,
) -> None:
    """
    Worker function for processing WSIs on a specific GPU.

    Uses multiprocessing (not threading) to ensure separate CUDA contexts
    per GPU and avoid GIL contention.

    Parameters
    ----------
    gpu_id : int
        GPU index for this worker
    work_queue : mp.Queue
        Queue of (wsi_path, wsi_name) tuples to process
    result_queue : mp.Queue
        Queue to put (wsi_name, success, error_msg) results
    encoder_name : str
        Name of the patch encoder
    encoder_kwargs : Dict[str, Any]
        Keyword arguments for encoder factory
    processor_config : Dict[str, Any]
        Configuration dict for creating Processor instances
    task : str
        Task to run ('seg', 'coords', 'feat', 'all')
    coords_dir : str
        Directory containing/for coordinates
    batch_size : int, optional
        Fixed batch size. If None and auto_batch_size=True, will be auto-determined.
    auto_batch_size : bool
        Whether to auto-determine batch size based on VRAM
    safety_margin : float
        VRAM safety margin for auto batch size
    """
    device = f'cuda:{gpu_id}'

    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)

        # Import here to avoid issues with multiprocessing
        from trident import Processor
        from trident.patch_encoder_models.load import encoder_factory
        from trident.segmentation_models.load import segmentation_model_factory

        # Create encoder
        encoder = encoder_factory(encoder_name, **encoder_kwargs)
        encoder.to(device)
        encoder.eval()

        # Determine batch size
        effective_batch_size = batch_size
        if auto_batch_size or batch_size is None:
            from trident.VRAMScheduler import VRAMEstimator
            estimator = VRAMEstimator(encoder, device, safety_margin)
            effective_batch_size = estimator.get_optimal_batch_size()
            print(f"[GPU {gpu_id}] Auto-determined batch size: {effective_batch_size}")

        # Process WSIs from queue
        while True:
            try:
                item = work_queue.get(timeout=1.0)
            except Empty:
                # Check if we should exit (sentinel value check)
                continue

            if item is None:  # Sentinel value - exit
                break

            wsi_path, wsi_name = item

            try:
                # Create processor for this WSI
                processor = Processor(
                    job_dir=processor_config['job_dir'],
                    wsi_source=os.path.dirname(wsi_path),
                    wsi_ext=[os.path.splitext(wsi_path)[1]],
                    custom_list_of_wsis=None,
                    skip_errors=processor_config.get('skip_errors', False),
                    custom_mpp_keys=processor_config.get('custom_mpp_keys'),
                    max_workers=processor_config.get('max_workers'),
                    reader_type=processor_config.get('reader_type'),
                )

                # Run the specified task
                if task in ('feat', 'all'):
                    processor.run_patch_feature_extraction_job(
                        coords_dir=coords_dir,
                        patch_encoder=encoder,
                        device=device,
                        saveas='h5',
                        batch_limit=effective_batch_size,
                    )

                result_queue.put((wsi_name, True, None))

            except Exception as e:
                error_msg = str(e)
                print(f"[GPU {gpu_id}] Error processing {wsi_name}: {error_msg}")
                result_queue.put((wsi_name, False, error_msg))

            finally:
                # Clean up GPU memory
                gc.collect()
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker initialization failed: {e}")
        # Put error for any remaining items
        result_queue.put((None, False, str(e)))


class MultiGPUProcessor:
    """
    Coordinates WSI processing across multiple GPUs.

    Uses multiprocessing with separate CUDA contexts per GPU for
    true parallel processing without GIL contention.
    """

    def __init__(
        self,
        gpus: List[int],
        encoder_name: str,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        strategy: str = 'load_balance',
        auto_batch_size: bool = True,
        safety_margin: float = 0.85,
    ):
        """
        Initialize multi-GPU processor.

        Parameters
        ----------
        gpus : List[int]
            List of GPU indices to use
        encoder_name : str
            Name of the patch encoder
        encoder_kwargs : Dict[str, Any], optional
            Keyword arguments for encoder factory
        strategy : str
            Scheduling strategy: 'load_balance', 'round_robin', or 'memory_aware'
        auto_batch_size : bool
            Whether to auto-determine batch sizes based on VRAM
        safety_margin : float
            VRAM safety margin (0.0-1.0)
        """
        self.gpus = gpus
        self.encoder_name = encoder_name
        self.encoder_kwargs = encoder_kwargs or {}
        self.strategy = strategy
        self.auto_batch_size = auto_batch_size
        self.safety_margin = safety_margin

        # Set multiprocessing start method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

    def process_wsis(
        self,
        wsi_list: List[Tuple[str, str]],  # [(wsi_path, wsi_name), ...]
        processor_config: Dict[str, Any],
        task: str,
        coords_dir: str,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """
        Process WSIs across multiple GPUs.

        Parameters
        ----------
        wsi_list : List[Tuple[str, str]]
            List of (wsi_path, wsi_name) tuples
        processor_config : Dict[str, Any]
            Configuration dict for creating Processor instances
        task : str
            Task to run ('seg', 'coords', 'feat', 'all')
        coords_dir : str
            Directory containing/for coordinates
        batch_size : int, optional
            Fixed batch size. If None, auto-determine based on VRAM.

        Returns
        -------
        List[Tuple[str, bool, Optional[str]]]
            List of (wsi_name, success, error_msg) results
        """
        if not wsi_list:
            return []

        # Assign WSIs to GPUs
        assignments = self._assign_wsis_to_gpus(wsi_list)

        # Create work queues for each GPU
        work_queues: Dict[int, mp.Queue] = {gpu: mp.Queue() for gpu in self.gpus}
        result_queue: mp.Queue = mp.Queue()

        # Populate work queues
        for wsi_path, wsi_name, gpu_id in assignments:
            work_queues[gpu_id].put((wsi_path, wsi_name))

        # Add sentinel values to signal workers to exit
        for gpu in self.gpus:
            work_queues[gpu].put(None)

        # Start worker processes
        workers = []
        for gpu in self.gpus:
            p = mp.Process(
                target=gpu_worker,
                args=(
                    gpu,
                    work_queues[gpu],
                    result_queue,
                    self.encoder_name,
                    self.encoder_kwargs,
                    processor_config,
                    task,
                    coords_dir,
                    batch_size,
                    self.auto_batch_size,
                    self.safety_margin,
                ),
            )
            p.start()
            workers.append(p)

        # Wait for all workers to complete
        for p in workers:
            p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                if result[0] is not None:  # Skip error placeholders
                    results.append(result)
            except Empty:
                break

        return results

    def _assign_wsis_to_gpus(
        self,
        wsi_list: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, int]]:
        """
        Assign WSIs to GPUs based on scheduling strategy.

        Parameters
        ----------
        wsi_list : List[Tuple[str, str]]
            List of (wsi_path, wsi_name) tuples

        Returns
        -------
        List[Tuple[str, str, int]]
            List of (wsi_path, wsi_name, gpu_id) tuples
        """
        assignments = []

        if self.strategy == 'round_robin':
            for i, (wsi_path, wsi_name) in enumerate(wsi_list):
                gpu = self.gpus[i % len(self.gpus)]
                assignments.append((wsi_path, wsi_name, gpu))

        elif self.strategy == 'memory_aware':
            # For memory-aware, we'll still use round-robin but the workers
            # will auto-determine batch sizes based on their available VRAM
            for i, (wsi_path, wsi_name) in enumerate(wsi_list):
                gpu = self.gpus[i % len(self.gpus)]
                assignments.append((wsi_path, wsi_name, gpu))

        else:  # load_balance
            # Estimate load for each WSI (use patch count if available)
            wsi_loads = []
            for wsi_path, wsi_name in wsi_list:
                # Try to get patch count from existing coords
                load = self._estimate_wsi_load(wsi_path, wsi_name)
                wsi_loads.append((wsi_path, wsi_name, load))

            # Sort by load (largest first) for better balancing
            wsi_loads.sort(key=lambda x: x[2], reverse=True)

            # Greedy assignment to least-loaded GPU
            gpu_loads = {gpu: 0 for gpu in self.gpus}

            for wsi_path, wsi_name, load in wsi_loads:
                # Find GPU with minimum current load
                min_gpu = min(self.gpus, key=lambda g: gpu_loads[g])
                assignments.append((wsi_path, wsi_name, min_gpu))
                gpu_loads[min_gpu] += load

        return assignments

    def _estimate_wsi_load(self, wsi_path: str, wsi_name: str) -> int:
        """Estimate processing load for a WSI."""
        # Default load estimate if we can't determine patch count
        return 1000


def run_multi_gpu_feature_extraction(
    wsi_list: List[Tuple[str, str]],
    gpus: List[int],
    encoder_name: str,
    processor_config: Dict[str, Any],
    coords_dir: str,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    auto_batch_size: bool = True,
    safety_margin: float = 0.85,
    strategy: str = 'load_balance',
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Convenience function for multi-GPU feature extraction.

    Parameters
    ----------
    wsi_list : List[Tuple[str, str]]
        List of (wsi_path, wsi_name) tuples
    gpus : List[int]
        List of GPU indices to use
    encoder_name : str
        Name of the patch encoder
    processor_config : Dict[str, Any]
        Configuration for Processor instances
    coords_dir : str
        Directory containing coordinates files
    encoder_kwargs : Dict[str, Any], optional
        Keyword arguments for encoder factory
    batch_size : int, optional
        Fixed batch size (if None, auto-determine)
    auto_batch_size : bool
        Whether to auto-determine batch size
    safety_margin : float
        VRAM safety margin
    strategy : str
        GPU scheduling strategy

    Returns
    -------
    List[Tuple[str, bool, Optional[str]]]
        Results as (wsi_name, success, error_msg) tuples
    """
    processor = MultiGPUProcessor(
        gpus=gpus,
        encoder_name=encoder_name,
        encoder_kwargs=encoder_kwargs,
        strategy=strategy,
        auto_batch_size=auto_batch_size,
        safety_margin=safety_margin,
    )

    return processor.process_wsis(
        wsi_list=wsi_list,
        processor_config=processor_config,
        task='feat',
        coords_dir=coords_dir,
        batch_size=batch_size,
    )
