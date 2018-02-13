# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import TakeFirst


def compact(s):
    """ returns None if string is empty, otherwise string itself """
    return s if s else None


class EwgScraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url = scrapy.Field(output_processor=TakeFirst())
    ingredient = scrapy.Field(output_processor=TakeFirst())
    ingredient_score = scrapy.Field(output_processor=TakeFirst())
    data_availability = scrapy.Field(output_processor=TakeFirst())
    overall_hazard_score = scrapy.Field(output_processor=TakeFirst())
    cancer_score = scrapy.Field(output_processor=TakeFirst())
    dev_reprod_tox_score = scrapy.Field(output_processor=TakeFirst())
    allergy_imm_tox_score = scrapy.Field(output_processor=TakeFirst())
    use_restrict_score = scrapy.Field(output_processor=TakeFirst())
