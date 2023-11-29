import sys


def debugger_is_active() -> bool:
    """
    Return if the debugger is currently active
    https://stackoverflow.com/a/67065084
    Tested on VSCode
    """
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
