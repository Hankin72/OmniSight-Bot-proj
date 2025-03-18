import base64
import configparser
import time
from datetime import datetime
import logging
import os
import re
import subprocess
import sys
import traceback
import numpy as np


class Color:
    """define color code"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'


def current_time():
    return "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] "

def printf(string):
    print(str(current_time()) + str(string))


def printWithColorCyan(string):
    print(Color.CYAN + str(current_time()) + string + Color.RESET)


def printWithColorGreen(string):
    print(Color.GREEN + str(current_time()) + string + Color.RESET)


def printWithColorRed(string):
    print(Color.RED + str(current_time()) + string + Color.RESET)


def printWithColorYellow(string):
    print(Color.YELLOW + str(current_time()) + string + Color.RESET)


def printWithColorBlue(string):
    print(Color.BLUE + str(current_time()) + string + Color.RESET)


def kill_process_on_port(port):
    cmd_find = "lsof -t -i:{}".format(port)
    process = subprocess.Popen(cmd_find, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    pid_str = output.decode("utf-8").strip()
    if pid_str:
        printWithColorCyan(f"Found process {pid_str} on port {port}, try to killing it now ...")
        try:
            os.kill(int(pid_str), 9)
            printWithColorCyan(f"Process {pid_str} killed successfully.")
        except ProcessLookupError:
            printWithColorRed(f"No process with PID {pid_str} found.")
    else:
        printWithColorYellow(f"No process running on port {port}")
        
        
def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度
    参数:
    - a: 向量 a
    - b: 向量 b
    返回:
    - similarity: 余弦相似度 (介于 -1 和 1 之间)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)