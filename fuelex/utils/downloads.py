import os
import time
import sys
import requests
import shutil
import threading

from queue import Queue
from collections import OrderedDict

class CacheItem:
    def __init__(self,id,content):
        self.id = id
        self.conent = content
        self.readers = 0

class DownloadCache:
    def __init__(self,cache_dir,cache_size):

        os.makedirs(cache_dir,exist_ok=True)
        self.cache_size = cache_size

        self.cache = OrderedDict()
        self.cache_lock =  threading.Lock()
        self.pending_inserts = []
        
    def checkout(self,id):
        content = None
        with self.cache_lock:
            if id in self.cache:
                content = self.cache[id].content
                self.cache[id].readers = self.cache[id].readers + 1
                self.cache.move_to_end(id,last=False)
        return content

    def release(self,item):
        with self.cache_lock:
            if item in self.cache:
                self.cache[item].readers = self.cache[item].readers - 1
    def poll(self,id):
        with self.cache_lock:
            result = id in self.cache
        return result   

    def queue_for_cache_insert(self,id,content):
        wait_stepped = self.attempt_pop_and_insert(id,content)
        while wait_stepped:
            wait_stepped = self.attempt_pop_and_insert(id,content)
            time.sleep(3)


    def insert(self,id,content):
        cleared = False
        queue_self = False

        while not cleared:
            with self.cache_lock:
                if id in self.cache:
                    cleared = True
                elif id in self.pending_inserts:
                    cleared = False
                else:
                    self.pending_inserts.append(id)
                    queue_self=True
                    cleared = True

            if queue_self:
                self.queue_for_cache_insert(id,content)
            elif not cleared:
                time.sleep(3)
                   

    def attempt_pop_and_insert(self,id,content):
        wait_stepped = True
        with self.cache_lock:
            next_removed = next(reversed(self.cache.values()))
            if next_removed.readers == 0:
                self.pop()
                self.cache[id] = CacheItem(id,content)
                wait_stepped=False
        return wait_stepped

    def pop(self):
        with self.cache_lock:
            self.cache.popitem(last=True)