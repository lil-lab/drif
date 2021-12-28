class TerminalColors:
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    ORANGE = '\033[33m'
    RED = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error(obj):
    print(TerminalColors.RED + str(obj) + TerminalColors.ENDC)


def print_warning(obj):
    print(TerminalColors.ORANGE + str(obj) + TerminalColors.ENDC)


def print_good(obj):
    print(TerminalColors.GREEN + str(obj) + TerminalColors.ENDC)