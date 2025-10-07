# lightweight package init: only export the async composite function
from .nifty_composite import analyze_nifty_composite

__all__ = ["analyze_nifty_composite"]
