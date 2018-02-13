# -*- coding: utf-8 -*-
import scrapy
from ewg_skindeep_uris import skindeep_uris
from scrapy import Selector
from ewg_scraper.items import EwgScraperItem
from scrapy.loader import ItemLoader
from scrapy.exceptions import CloseSpider
from scrapy.shell import inspect_response
import urlparse


class EwgSkindeepSpider(scrapy.Spider):
    name = "ewg_skindeep"
    allowed_domains = ["ewg.org"]
    start_urls = ['http://www.ewg.org' + uri for uri in skindeep_uris]

    # Constants (these might need updates if the site is redisigned)
    ovrll_hz_ps, cncr_ps, dv_rprd_tx_ps, allrgy_imm_tx_ps, use_rstrct_ps = (0, 1, 2, 3, 4)
    product_link_xpath = '//td[@class="product_name_list"]/a/@href'
    ingredient_link_xpath = (
        '//div[@id="Ingredients"]/table[@class="product_tables"]'
        '//td[@class="firstcol"]/a/@href')
    ingredient_xpath = '//div[@id="righttoptitleandcats"]/h1/text()'
    ingredient_score_xpath = '//div[@class="scoretextbg"]//img/@src'
    data_availability_xpath = '//div[@class="purple2012"]/span[2]/text()'
    score_bars_xpath = '//div[@class="basic_bar"]/@style'

    # Lists for keeping track of crawler status
    crawledCategoryUrls = []
    crawledProductUrls = []
    crawledIngredientUrls = []
    savedSongs = []
    numCrawled = 0

    def try_cast(self, value, type=int):
        # Try to cast value to input type return casted value or None if unsuccessful
        ret = None
        try:
            ret = type(value)
        except ValueError:
            pass
        return ret

    def parse(self, response):
        # Parse a product group page
        #self.logger.info('[parse] Called on %s', response.url)
        self.crawledCategoryUrls.append(response.url)  # Mark category page as seen

        # Get HTML response
        responseHTML = response.body
        sel = Selector(text=responseHTML, type="html")
        found_product_links = False

        # Find all product links in the page and add them to the list for crawling
        # This prevents the spider from crawling the same link multiple times or looping
        for href in sel.xpath(self.product_link_xpath).extract():
            if not href in self.crawledProductUrls:
                self.crawledProductUrls.append(href)
                product_url = urlparse.urljoin(response.url, href)
                found_product_links = True
                #self.logger.info('[parse] Found product link: %s', product_url)
                yield scrapy.Request(product_url, callback=self.parse_product)
        if not found_product_links:
            pass
            #self.logger.warning('[parse] Could not extract product links from: %s', response.url)

    def parse_product(self, response):
        # Parse a product page
        #self.logger.info('[parse_product] Called on %s', response.url)
        self.crawledProductUrls.append(response.url)
        responseHTML = response.body
        sel = Selector(text=responseHTML, type="html")
        found_ingredient_links = False
        for href in sel.xpath(self.ingredient_link_xpath).extract():
            ingredient_url = urlparse.urljoin(response.url, href)
            if not ingredient_url in self.crawledIngredientUrls:
                self.crawledIngredientUrls.append(ingredient_url)
                found_ingredient_links = True
                yield scrapy.Request(ingredient_url, callback=self.parse_ingredient)
        if not found_ingredient_links:
            pass
            #self.logger.warning('[parse_product] Could not extract ingredient links from: %s', response.url)

    def parse_ingredient(self, response):
        # Parse ingredient page
        self.logger.info('[parse_ingredient] Called on %s', response.url)
        l = ItemLoader(item=EwgScraperItem(), response=response)

        # Get ingredient score from image filename
        score_img_uri = response.xpath(self.ingredient_score_xpath).extract_first()
        score_img_name = score_img_uri[score_img_uri.rfind("/")+1:]
        ingredient_score = self.try_cast(
            score_img_name[:score_img_name.find(".")].replace("score_image", "")[0])

        # Get score bar values
        scr_brs = response.xpath(self.score_bars_xpath)
        overall_hazard_score = self.try_cast(
            scr_brs[self.ovrll_hz_ps].extract().replace("width:", "").replace("px", ""),
            float)
        cancer_score = self.try_cast(
            scr_brs[self.cncr_ps].extract().replace("width:", "").replace("px", ""),
            float)
        dev_reprod_tox_score = self.try_cast(
            scr_brs[self.dv_rprd_tx_ps].extract().replace("width:", "").replace("px", ""),
            float)
        allergy_imm_tox_score = self.try_cast(
            scr_brs[self.allrgy_imm_tx_ps].extract().replace("width:", "").replace("px", ""),
            float)
        use_restrict_score = self.try_cast(
            scr_brs[self.use_rstrct_ps].extract().replace("width:", "").replace("px", ""),
            float)

        l.add_value('url', response.url)
        l.add_xpath('ingredient', self.ingredient_xpath)
        l.add_value('ingredient_score', ingredient_score)
        l.add_xpath('data_availability', self.data_availability_xpath)
        l.add_value('overall_hazard_score', overall_hazard_score)
        l.add_value('cancer_score', cancer_score)
        l.add_value('dev_reprod_tox_score', dev_reprod_tox_score)
        l.add_value('allergy_imm_tox_score', allergy_imm_tox_score)
        l.add_value('use_restrict_score', use_restrict_score)

        item = l.load_item()
        self.crawledIngredientUrls.append(response.url)
        self.logger.info('[parse_ingredient] Added info for %s', item['ingredient'])
        #inspect_response(response, self)
        yield item
