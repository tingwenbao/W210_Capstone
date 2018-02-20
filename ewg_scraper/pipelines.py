# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy import signals
import json
from scrapy.utils.serialize import ScrapyJSONEncoder
from RepeatedTimer import RepeatedTimer
encoder = ScrapyJSONEncoder()


class EwgScraperPipeline(object):

    def __init__(self):
        self.item_dict = {}
        self.ingredientsCrawled = 0
        self.productsCrawled = 0
        print "starting..."
        self.rt = RepeatedTimer(1, self.update_stats)  # it auto-starts, no need of rt.start()

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_closed(self, spider):
        self.rt.stop()
        with open('%s_ingredients.json' % spider.name, 'w') as f:
            json.dump([entry for entry in self.item_dict.values()], f)
        print "Crawled {} ingredients".format(spider.ingredientsCrawled)

    def process_item(self, item, spider):
        # Add ingredient or product to collected data
        # NOTE: Does nothing if input item is not a product or ingredient
        if item:
            is_ingredient = 'ingredient_id' in item.keys()
            item_key = 'ingredient_id' if is_ingredient else 'product_id'
            if 'product_id' in item.keys() or is_ingredient:
                input_dict = dict(item)
                input_key = input_dict[item_key]
                if input_key in self.item_dict:
                    # If item already exists make sure the new input is not another placeholder
                    # NOTE: A place holder item contains only the item id
                    if len(input_dict) > 1:
                        self.item_dict[input_key].update(input_dict)
                else:
                    self.item_dict[input_key] = input_dict
                if is_ingredient:
                    self.ingredientsCrawled = self.ingredientsCrawled + 1
                else:
                    self.productsCrawled = self.productsCrawled + 1

    def update_stats(self):
        print "Ingredients: {}\tProducts: {}".format(self.ingredientsCrawled, self.productsCrawled)
