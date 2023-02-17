from dataclasses import dataclass
import sys, smtplib, json, urllib3
import matplotlib.pyplot as plt
from datetime import datetime
from pytrends.request import TrendReq
import numpy as np
import pandas as pd
import yfinance as yf
import tqdm.notebook as tq
urllib3.disable_warnings()
