"""Alias the package downloader so legacy imports share its mutable state."""

import sys

from aigc_detector import network

sys.modules[__name__] = network
