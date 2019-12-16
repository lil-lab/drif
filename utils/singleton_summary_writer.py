from utils.logging_summary_writer import LoggingSummaryWriter
import parameters.parameter_server as P

SUMMARY_WRITER = None

def get():
    global SUMMARY_WRITER
    if SUMMARY_WRITER is None:
        run_name = P.get_current_parameters()["Setup"].get("run_name") or "untitled_run"
        SUMMARY_WRITER = LoggingSummaryWriter(log_dir="runs/" + run_name)
    return SUMMARY_WRITER