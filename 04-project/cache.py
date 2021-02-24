import logging
from typing import Text

from diskcache import Disk, FanoutCache

log = logging.getLogger(__name__)


def get_cache(scope_str: Text) -> FanoutCache:
    return FanoutCache(
        'cache/' + scope_str,
        disk=Disk,
        shards=64,
        timeout=1,
        size_limit=3e11,
    )
