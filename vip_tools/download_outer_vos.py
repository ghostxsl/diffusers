# Copyright (c) wilson.xu. All rights reserved.
import os
from os.path import basename, join
import argparse
import threading
import requests
import time
from tqdm import tqdm
import random

from diffusers.data.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="llm-vos upload script.")
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        help=" ")
    parser.add_argument(
        "--save_dir",
        default="output",
        type=str,
        help="Directory to save.")
    parser.add_argument(
        "--num_thread",
        default=2,
        type=int,
        help="The number of multithreading.")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    return args


class ImageDownloader(object):
    def __init__(self, num_thread=0):
        self.num_thread = num_thread
        self.sleep_interval = 1000
        self.sleep_time = 1  # second
        self.hosts = ['http://a.vpimg1.com/url', 'https://a.vpimg2.com']

        self.failed_list = []

    def _download_worker(self, thread_i, urls, save_paths):
        assert len(urls) == len(save_paths)
        cnt = 0
        for url, path in tqdm(zip(urls, save_paths), desc=f'thread-{thread_i}'):
            if cnt % self.sleep_interval == 0:
                time.sleep(self.sleep_time * random.random())
            cnt += 1
            try:
                self._get_img(url, path)
            except Exception as e:
                print(e)

    def _get_img(self, url, save_path):
        download_success = False
        try:
            for host in self.hosts:
                url = url if url.startswith('http') else join(host, url.lstrip('/'))

                res = requests.get(url, timeout=3)
                if res.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(res.content)

                    download_success = True
                    break
                else:
                    print(res.status_code)
        except Exception as e:
            print(e)

        if not download_success:
            self.failed_list.append(url)

        return download_success

    def download_images(self, data_list, save_dir):

        save_list = []
        for url in data_list:
            name = basename(url)
            save_list.append(join(save_dir, name))

        if self.num_thread == 0:
            for url, save_path in tqdm(zip(data_list, save_list)):
                self._get_img(url, save_path)
        else:
            threads = []
            for i in range(self.num_thread):
                thread = threading.Thread(
                    target=self._download_worker,
                    args=[i, data_list[i::self.num_thread], save_list[i::self.num_thread]])
                threads.append(thread)

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        return save_list


if __name__ == "__main__":
    args = parse_args()

    data_list = load_file(args.data_file)
    print(len(data_list))

    downloader = ImageDownloader(num_thread=args.num_thread)
    downloader.download_images(data_list, args.save_dir)

    print('Done!')
    print(downloader.failed_list)
    json_save(downloader.failed_list, "./failed_list.json")
