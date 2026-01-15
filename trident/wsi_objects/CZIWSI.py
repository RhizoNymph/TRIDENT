from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional

from trident.wsi_objects.WSI import WSI, ReadMode


class CZIWSI(WSI):
    """
    WSI reader for Carl Zeiss CZI files using pylibCZIrw.
    """

    def __init__(self, slide_path: str, **kwargs) -> None:
        """
        Initialize a WSI object from a CZI file.

        Parameters
        ----------
        slide_path : str
            Path to the CZI file.
        mpp : float, optional
            Microns per pixel. If not provided, will be extracted from CZI metadata.
        name : str, optional
            Optional name for the slide.
        lazy_init : bool, default=True
            Whether to defer initialization until the WSI is accessed.
        """
        try:
            import pylibCZIrw.czi as pyczi
        except ImportError:
            raise ImportError(
                "pylibCZIrw is required for CZI files. "
                "Install with: pip install pylibCZIrw"
            )

        self._czi = None
        self._czi_context = None
        self._bbox = None
        self._bbox_tuple = None  # For pickling

        # Try to get MPP from kwargs or extract from file
        if kwargs.get("mpp") is None:
            kwargs["mpp"] = self._extract_mpp_from_file(slide_path)

        super().__init__(slide_path, **kwargs)

    def _extract_mpp_from_file(self, slide_path: str) -> Optional[float]:
        """Extract MPP from CZI file metadata."""
        try:
            import pylibCZIrw.czi as pyczi
            with pyczi.open_czi(slide_path) as czi:
                meta = czi.metadata
                scaling = meta['ImageDocument']['Metadata']['Scaling']['Items']['Distance']
                for dim in scaling:
                    if dim['@Id'] == 'X':
                        # Value is in meters, convert to microns
                        return float(dim['Value']) * 1e6
        except Exception:
            pass
        return None

    def _lazy_initialize(self) -> None:
        """Lazily initialize the WSI from a CZI file."""
        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                self._ensure_czi_open()

                # Get bounding box for dimensions
                bbox = self._czi.total_bounding_box
                self._bbox_tuple = (bbox['X'][0], bbox['Y'][0],
                                    bbox['X'][1] - bbox['X'][0],
                                    bbox['Y'][1] - bbox['Y'][0])

                self.width = self._bbox_tuple[2]
                self.height = self._bbox_tuple[3]
                self.dimensions = (self.width, self.height)

                # CZI files are single-level (we read at full resolution)
                self.level_downsamples = [1]
                self.level_dimensions = [(self.width, self.height)]
                self.level_count = 1

                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True

            except Exception as e:
                raise Exception(f"Error initializing CZI WSI: {e}")

    def _ensure_czi_open(self):
        """Ensure the CZI file is open."""
        if self._czi is None:
            import pylibCZIrw.czi as pyczi
            # open_czi returns a context manager, we need to enter it
            self._czi_context = pyczi.open_czi(self.slide_path)
            self._czi = self._czi_context.__enter__()
            # Get bbox if not cached
            if self._bbox_tuple is None:
                bbox = self._czi.total_bounding_box
                self._bbox_tuple = (bbox['X'][0], bbox['Y'][0],
                                    bbox['X'][1] - bbox['X'][0],
                                    bbox['Y'][1] - bbox['Y'][0])

    def __getstate__(self):
        """Prepare state for pickling - close file handle."""
        state = self.__dict__.copy()
        # Don't pickle the CZI file object
        state['_czi'] = None
        state['_czi_context'] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._czi = None
        self._czi_context = None

    def get_dimensions(self):
        return self.dimensions

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """Generate a thumbnail of the CZI image."""
        self._ensure_czi_open()

        # Calculate scale factor for thumbnail
        scale_x = size[0] / self.width
        scale_y = size[1] / self.height
        scale = min(scale_x, scale_y)

        # Read at reduced resolution
        x, y, w, h = self._bbox_tuple
        mosaic = self._czi.read(
            roi=(x, y, w, h),
            zoom=scale,
            plane={'C': 0}
        )

        # Convert to PIL Image
        mosaic = self._convert_to_rgb(mosaic)
        img = Image.fromarray(mosaic)
        img.thumbnail(size)
        return img.convert('RGB')

    def _convert_to_rgb(self, mosaic: np.ndarray) -> np.ndarray:
        """Convert CZI array format to RGB HWC uint8."""
        # Handle various array formats from pylibCZIrw
        if mosaic.ndim == 4:
            mosaic = mosaic[0, :, :, :]  # Remove batch/Z dimension
        if mosaic.ndim == 3 and mosaic.shape[0] in [1, 3, 4]:
            mosaic = np.moveaxis(mosaic, 0, -1)  # CHW -> HWC

        # Ensure 3 channels
        if mosaic.ndim == 2:
            mosaic = np.stack([mosaic, mosaic, mosaic], axis=-1)
        elif mosaic.shape[-1] == 1:
            mosaic = np.repeat(mosaic, 3, axis=-1)
        elif mosaic.shape[-1] == 4:
            mosaic = mosaic[:, :, :3]  # RGBA -> RGB

        return mosaic.astype(np.uint8)

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """Extract a specific region from the CZI file."""
        if level != 0:
            raise ValueError("CZIWSI only supports reading at level=0.")

        self._ensure_czi_open()

        # Adjust location relative to bounding box origin
        bbox_x, bbox_y, _, _ = self._bbox_tuple
        x = location[0] + bbox_x
        y = location[1] + bbox_y
        w, h = size

        try:
            # Read the region using pylibCZIrw
            mosaic = self._czi.read(
                roi=(x, y, w, h),
                zoom=1.0,
                plane={'C': 0}
            )
        except RuntimeError as e:
            # Handle corrupt tiles by returning a blank region
            if 'WMP_errFail' in str(e) or 'ERR=-1' in str(e):
                # Return a white/blank tile
                mosaic = np.ones((h, w, 3), dtype=np.uint8) * 255
            else:
                raise

        mosaic = self._convert_to_rgb(mosaic)

        # Ensure correct size (CZI might return slightly different size)
        if mosaic.shape[0] != h or mosaic.shape[1] != w:
            img = Image.fromarray(mosaic).resize((w, h), Image.Resampling.BILINEAR)
            mosaic = np.array(img)

        if read_as == 'pil':
            return Image.fromarray(mosaic).convert('RGB')
        elif read_as == 'numpy':
            return mosaic
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def segment_tissue(self, *args, **kwargs):
        out = super().segment_tissue(*args, **kwargs)
        self.close()
        return out

    def extract_tissue_coords(self, *args, **kwargs):
        out = super().extract_tissue_coords(*args, **kwargs)
        self.close()
        return out

    def visualize_coords(self, *args, **kwargs):
        out = super().visualize_coords(*args, **kwargs)
        self.close()
        return out

    def extract_patch_features(self, *args, **kwargs):
        out = super().extract_patch_features(*args, **kwargs)
        self.close()
        return out

    def extract_slide_features(self, *args, **kwargs):
        out = super().extract_slide_features(*args, **kwargs)
        self.close()
        return out

    def close(self):
        """Close the CZI file to free resources."""
        if self._czi_context is not None:
            try:
                self._czi_context.__exit__(None, None, None)
            except:
                pass
            self._czi = None
            self._czi_context = None
