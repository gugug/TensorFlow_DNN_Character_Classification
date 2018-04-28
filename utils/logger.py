# coding=utf-8
__author__ = 'gu'
import os
import logging


def get_logger():
    logging.basicConfig(filename=os.path.join('' '../log.txt'),
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.INFO)
    # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging
