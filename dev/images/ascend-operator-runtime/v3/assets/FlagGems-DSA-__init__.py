"""Package marker for FlagGems fused.DSA.

The locked FlagGems commit imports this namespace from fused/__init__.py but
does not include an __init__.py, while its setuptools build uses find_packages.
Keeping this marker in the derived-image layer makes those imported modules
part of the installed wheel without modifying the FlagGems source checkout.
"""
