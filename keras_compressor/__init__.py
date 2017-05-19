from .layers import custom_layers

__all__ = ['custom_objects']
custom_objects = dict(custom_layers.items())  # shallow copy
