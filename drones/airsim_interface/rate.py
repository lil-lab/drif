from time import sleep
from datetime import datetime


class Rate:
    """
    Class that sleeps the appropriate amount, such that the interval between subsequent
    calls to Rate.sleep() is always the desired "rate".
    #TODO: Rename rate argument to interval, since that's what it really is
    """

    def __init__(self, rate):
        self.rate = rate
        self.last_ticks = self.now()

    def now(self):
        datetime_now = datetime.now()
        secs = \
            datetime_now.month * 12*24*60*60 + \
            datetime_now.day * 24*60*60 + \
            datetime_now.hour * 60 * 60 + \
            datetime_now.minute * 60 + \
            datetime_now.second + datetime_now.microsecond * 1e-6

        return secs

    def reset(self):
        self.last_ticks = self.now()

    def sleep(self, quiet=False):
        now = self.now()
        next_time = self.last_ticks + self.rate
        self.last_ticks = now
        sleep_time = next_time - now
        if sleep_time <= 0:
            if not quiet:
                print("Warning: Rate missed desired interval: " + str(self.rate) + " by " + str(sleep_time))
            return
        sleep(sleep_time)
        #Uncomment for debugging:
        #after = self.now()
        #print("Rate sleep duration: ", after - now, sleep_time)

    def sleep_n_intervals(self, n):
        self.reset()
        for i in range(n):
            self.sleep(quiet=True)
