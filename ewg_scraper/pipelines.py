# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy import signals
import json
from scrapy.utils.serialize import ScrapyJSONEncoder
encoder = ScrapyJSONEncoder()


class EwgScraperPipeline(object):

    def __init__(self):
        self.items = []

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_closed(self, spider):
        with open('%s_ingredients.json' % spider.name, 'w') as f:
            json.dump(self.items, f)
        print "Crawled {} ingredients".format(spider.itemsCrawled)

    def process_item(self, item, spider):
        # Require an ingredient name and score at a minimum
        if item:
            if 'ingredient' in item.keys() and 'ingredient_score' in item.keys():
                self.items.append(dict(item))
        if not spider.itemsCrawled % 100:
            print "Crawled {} ingredients".format(spider.itemsCrawled)
