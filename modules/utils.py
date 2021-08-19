# Process management
import os, sys
import code, traceback, signal

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def register_debug_hook():
    signal.signal(signal.SIGUSR1, debug)  # Register handler

def register_kill_hook():
    def _exit(signum, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, _exit)
    signal.signal(signal.SIGTERM, _exit)

# nn model loading
import torch
from modules.hyperparameters.randomsearch import generate_ann

def load_neural_model(directory: str, device: torch.device=torch.device('cpu')):
    meta_data = torch.load(directory + '/_meta.pt')

    nn_parameters = torch.load(directory + '/_parameters.pt', map_location=device)
    nn_model = generate_ann(nn_parameters)
    nn_model.to(device)
    nn_model.load_state_dict(torch.load(directory + '/_nn.pt', map_location=device))

    return meta_data, nn_parameters, nn_model

# math
import numpy as np

def discretize_value(value, discrete_values):
    return discrete_values[np.argmin(np.abs(value - discrete_values))]

# Logging
import logging

def process_log_records(queue):
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record) 