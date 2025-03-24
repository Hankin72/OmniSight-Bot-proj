from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import threading
import os
import time
import numpy as np

