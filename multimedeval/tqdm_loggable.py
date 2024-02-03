"""Progress-to-logs tqdm implementation."""

import logging
from datetime import datetime, timedelta

from tqdm.auto import tqdm as tqdm_auto




class tqdm_logging(tqdm_auto):
    """A tqdm implementation that outputs to Python logger.

    - Any postfix progress bar arguments are passed as `extras` to the logging system

    - The log has structured data available; all logs are tagged with `{"tqdm": True}`
      so they can be filtered out in the log management service

    - Default rate of creating messages is lower than the usually progress bar
      update rate of 1 Htz, because we do not want to spam the logs

    See also

    - `Implementing structured logging <https://docs.python.org/3/howto/logging-cookbook.html#implementing-structured-logging>`_.
    """



    #: What log level all tqdm_logging instances will use
    log_level = logging.INFO


    #: How often to post a log message
    #: Default to every 10 seconds
    log_message_rate = timedelta(seconds=10)

    def __init__(self, logger, *args, **kwargs):
        self.last_log_message_at = datetime(1970, 1, 1)
        self.logger:logging.Logger = logger
        super(tqdm_logging, self).__init__(*args, **kwargs)



    @classmethod
    def set_level(cls, log_level: int):
        """Set log level to all tqdm_logging instances.

        Currently we do not support per-instance logging level
        to maintain argument compatibility with std.tqdm constructor.
        """
        cls.log_level = log_level

    @classmethod
    def set_log_rate(cls, rate: timedelta):
        """Set the rate how often we post a log message."""
        assert isinstance(rate, timedelta)
        cls.log_message_rate = rate

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """Overloaded to store the raw post-fix"""
        self.raw_postfix = ordered_dict
        super(tqdm_logging, self).set_postfix(ordered_dict, refresh, **kwargs)

    def should_throttle_logging(self) -> bool:
        """Check if we throttle down displays to reduce log spam"""

        # Always refresh the last message on tqdm.close()
        if self.disable:
            return False

        if datetime.now() - self.last_log_message_at > self.log_message_rate:
            return False
        else:
            return True

    def display(self, **kwargs):
        """Create a log entry for the current progress."""

        if self.should_throttle_logging():
            return

        # Some sample format_dicts:
        #
        # {'n': 0, 'total': 60, 'elapsed': 0, 'ncols': 344, 'nrows': 15, 'prefix': 'Sample progress', 'ascii': False, 'unit': 'it', 'unit_scale': False, 'rate': None, 'bar_format': None, 'postfix': None, 'unit_divisor': 1000, 'initial': 0, 'colour': None}
        # {'n': 7, 'total': 60, 'elapsed': 3.032935857772827, 'ncols': 344, 'nrows': 15, 'prefix': 'Sample progress', 'ascii': False, 'unit': 'it', 'unit_scale': False, 'rate': 2.0927977450656536, 'bar_format': None, 'postfix': 'Currently time=2022-09-20 21:08:26.951320', 'unit_divisor': 1000, 'initial': 0, 'colour': None}
        # ^C{'n': 3000, 'total': 60000, 'elapsed': 1.089920997619629, 'ncols': 344, 'nrows': 15, 'prefix': 'Sample progress', 'ascii': False, 'unit': 'it', 'unit_scale': True, 'rate': None, 'bar_format': None, 'postfix': 'Currently time=2022-09-20 21:10:24.377513', 'unit_divisor': 1000, 'initial': 0, 'colour': None}
        #
        format_dict = self.format_dict
        name = format_dict.get("prefix", "unknown")
        postfix = format_dict.get("postfix", None)
        rate = format_dict.get("rate", 0)
        unit = format_dict.get("unit", None)
        unit_divisor = format_dict.get("unit_divisor", None)
        elapsed = format_dict.get("elapsed", 0)
        n = format_dict.get("n", 0)
        total = format_dict.get("total", -1)

        if n and total:
            n_formatted = self.format_sizeof(n, unit, unit_divisor)
            total_formatted = self.format_sizeof(total, unit, unit_divisor)
        else:
            # Progress bar without total
            n_formatted = n
            total_formatted = total

        n_formatted = n_formatted or "-"
        total_formatted = total_formatted or "-"

        # Taken from format_meter()
        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm_auto.format_interval(remaining) if rate else '?'
        try:
            eta_dt = (datetime.now() + timedelta(seconds=remaining)
                      if rate and total else datetime.utcfromtimestamp(0))
        except OverflowError:
            eta_dt = datetime.max

        elapsed_str = tqdm_auto.format_interval(elapsed)

        if rate:
            if rate > 1:
                rate_formatted = f"{rate:,.1f}{unit}/s"
            else:
                rate_formatted = f"{1/rate:,.1f}s/{unit}"
        else:
            rate_formatted = "-"

        # Include any postfix variables in extra logging
        raw_postfix = getattr(self, "raw_postfix", {})

        postfix_str = postfix or "-"

        # Structured log to be passed to Sentry / LogStash
        structured_logs = {
            "tqdm": True,
            "progress_name": name,
            "rate": rate,
            "unit": unit,
            "elapsed": elapsed,
            "n": n,
            "total": total,
            # Ideally we'd like to submit raw postfix so structure is retained, but
            # it may contain non-JSON'able values that choke some  loggers
            # e.g. python-logstash
            "postfix": postfix_str,
            "eta": eta_dt.isoformat(),
            "remaining": remaining,
        }


        self.logger.log(
            self.log_level,
            self.__str__(),
            # name,
            # n_formatted,
            # total_formatted,
            # rate_formatted,
            # remaining_str,
            # elapsed_str,
            # postfix_str,
            extra=structured_logs,
        )

        self.last_log_message_at = datetime.now()

    def close(self):
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        self.display()