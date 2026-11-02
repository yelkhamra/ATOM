"""The served geometry: the GDN recurrent state stored fp32 (``Geometry.gdn``), as ATOM's state pool keeps it."""

from atom.mono.program.geometry import GDNGeometry
from atom.models.qwen4_exp_mono.whole_layer.config import G, Geometry


def served_geometry(geo: Geometry = G) -> Geometry:
    return geo


def gdn_geometry(geo: Geometry = G) -> GDNGeometry:
    """The GDN geometry the kernels and ATOM's pools run with."""
    return geo.gdn
