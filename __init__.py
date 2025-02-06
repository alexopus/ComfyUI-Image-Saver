from .nodes import NODE_CLASS_MAPPINGS as NODES_OLD
from .civitai_nodes import NODE_CLASS_MAPPINGS as NODES_NEW

# Merge both mappings
NODE_CLASS_MAPPINGS = {**NODES_OLD, **NODES_NEW}

__all__ = ['NODE_CLASS_MAPPINGS']
