import sys
class Logger:
    def __init__(self, level=3):
        self.level = level
    def clear_line(self, num: int = 1):
        # sys.stdout.write("\033[K")
        sys.stdout.write("\033[F")
    def info(self, msg: str):
        if (self.level >= 3):
            print("[INFO]", msg)
    def warn(self, msg: str):
        if (self.level >= 2): print("[WARN]", msg)
    def error(self, msg: str):
        if self.level >= 1: print("[ERROR]", msg)