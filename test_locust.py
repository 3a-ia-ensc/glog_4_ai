from os import path
import pandas as pd
from random import randint
import urllib

import time
from locust import HttpUser, task, between


class QuickstartUser(HttpUser):
    wait_time = between(0.5, 1)

    @task
    def index_page(self):
        df_data = pd.read_json(path.join('data', 'testing_set.json'))
        idx = randint(0, len(df_data))
        rnd_sentence = str(df_data['sentence'][idx])
        s = urllib.parse.quote(rnd_sentence, safe='')
        self.client.get(f'/api/intent?sentence={s}')
