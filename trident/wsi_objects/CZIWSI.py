from __future__ import annotations
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional

from trident.wsi_objects.WSI import WSI, ReadMode


class CZIWSI(WSI):
    """
    WSI reader for Carl Zeiss CZI files using aicspylibczi.
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

        Examples
        --------
        >>> wsi = CZIWSI("path/to/image.czi", lazy_init=False)
        >>> print(wsi)
        <width=5120, height=3840, backend=CZIWSI, mpp=0.11, mag=40>
        """
        try:
            from aicspylibczi import CziFile
        except ImportError:
            raise ImportError(
                "aicspylibczi is required for CZI files. "
                "Install with: pip install 'aicspylibczi>=3.1.1'"
            )

        self._czi = None
        self._czi_class = CziFile
        self._bbox = None

        # Try to get MPP from kwargs or extract from file
        if kwargs.get("mpp") is None:
            kwargs["mpp"] = self._extract_mpp_from_file(slide_path)

        super().__init__(slide_path, **kwargs)

    def _extract_mpp_from_file(self, slide_path: str) -> Optional[float]:
        """Extract MPP from CZI file metadata using pylibCZIrw."""
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
        """
        Lazily initialize the WSI from a CZI file.
        """
        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                self._ensure_czi_open()

                # Get bounding box for dimensions
                self._bbox = self._czi.get_mosaic_bounding_box()

                self.width = self._bbox.w
                self.height = self._bbox.h
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
            self._czi = self._czi_class(self.slide_path)
            self._bbox = self._czi.get_mosaic_bounding_box()

    def get_dimensions(self):
        return self.dimensions

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the CZI image.

        Parameters
        ----------
        size : tuple of int
            Desired thumbnail size (width, height).

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail image.
        """
        self._ensure_czi_open()

        # Calculate scale factor for thumbnail
        scale_x = size[0] / self.width
        scale_y = size[1] / self.height
        scale = min(scale_x, scale_y)

        # Read scaled mosaic
        mosaic, _ = self._czi.read_mosaic(
            region=(self._bbox.x, self._bbox.y, self._bbox.w, self._bbox.h),
            scale_factor=scale,
            C=0
        )

        # Convert to PIL Image (mosaic is typically ZCYX or similar)
        if mosaic.ndim == 4:
            mosaic = mosaic[0, :, :, :]  # Remove Z dimension if present
        if mosaic.ndim == 3 and mosaic.shape[0] in [1, 3, 4]:
            mosaic = np.moveaxis(mosaic, 0, -1)  # CHW -> HWC

        if mosaic.shape[-1] == 1:
            mosaic = np.repeat(mosaic, 3, axis=-1)
        elif mosaic.shape[-1] == 4:
            mosaic = mosaic[:, :, :3]  # RGBA -> RGB

        img = Image.fromarray(mosaic.astype(np.uint8))
        img.thumbnail(size)
        return img.convert('RGB')

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the CZI file.

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
        level : int
            Pyramid level to read from. Only level 0 is supported.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        read_as : {'pil', 'numpy'}, optional
            Output format for the region.

        Returns
        -------
        Union[PIL.Image.Image, np.ndarray]
            Extracted image region.
        """
        if level != 0:
            raise ValueError("CZIWSI only supports reading at level=0.")

        self._ensure_czi_open()

        # Adjust location relative to bounding box origin
        x = location[0] + self._bbox.x
        y = location[1] + self._bbox.y
        w, h = size

        # Read the region
        mosaic, _ = self._czi.read_mosaic(
            region=(x, y, w, h),
            scale_factor=1.0,
            C=0
        )

        # Convert array format (ZCYX -> HWC)
        if mosaic.ndim == 4:
            mosaic = mosaic[0, :, :, :]  # Remove Z
        if mosaic.ndim == 3 and mosaic.shape[0] in [1, 3, 4]:
            mosaic = np.moveaxis(mosaic, 0, -1)  # CHW -> HWC

        # Ensure RGB
        if mosaic.shape[-1] == 1:
            mosaic = np.repeat(mosaic, 3, axis=-1)
        elif mosaic.shape[-1] == 4:
            mosaic = mosaic[:, :, :3]

        mosaic = mosaic.astype(np.uint8)

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
        if self._czi is not None:
            self._czi = None
