
logging_config = { 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            #'filename': '',
            'mode': 'w',
        }
    },
    'loggers': { 
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    } 
}