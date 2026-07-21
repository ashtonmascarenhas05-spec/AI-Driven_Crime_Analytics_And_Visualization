import numpy as np
import pandas as pd

class Scraper:

    def __init__(self,csv_file):
        self.csv_file = csv_file

    def pull_data(self,csv_file):
        file_data = pd.read_csv(self.csv_file)
        return file_data


## further scraping required           
            