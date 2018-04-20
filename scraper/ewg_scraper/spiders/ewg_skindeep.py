# -*- coding: utf-8 -*-
import scrapy
from ewg_skindeep_uris import category_params1, category_params2, category_params3
from scrapy import Selector
from ewg_scraper.items import EwgScraperIngredient, EwgScraperProduct
from scrapy.loader import ItemLoader
from scrapy.shell import inspect_response
import urlparse
from base64 import urlsafe_b64encode, urlsafe_b64decode


class EwgSkindeepSpider(scrapy.Spider):
    name = "ewg_skindeep"
    allowed_domains = ["ewg.org"]
    num_per_page = 5000
    req_params = "&&showmore=products&atatime=" + str(num_per_page)
    site_url = 'http://www.ewg.org/skindeep/browse.php?category='

    # Constants (these might need updates if the site is redisigned)
    ovrll_hz_ps, cncr_ps, dv_rprd_tx_ps, allrgy_imm_tx_ps, use_rstrct_ps = (0, 1, 2, 3, 4)
    product_tr_list_xpath = '//table[@id="table-browse"]//tr'
    product_type_xpath = 'td[4]/text()'  # Relative to product_tr elem
    product_link_xpath = 'td[@class="product_name_list"]/a/@href'  # Relative to product_tr elem

    ingredient_link_xpath = (
        '//div[@id="Ingredients"]/table[@class="product_tables"]'
        '//td[@class="firstcol"]/a/@href')
    name_xpath = '//div[@id="righttoptitleandcats"]/h1/text()'
    score_xpath = '//div[@class="scoretextbg"]//img/@src'
    data_availability_xpath = '//div[@class="purple2012"]/span[2]/text()'
    score_bars_xpath = '//div[@class="basic_bar"]/@style'

    # Lists for keeping track of crawler status
    crawledCategoryUrls = []
    crawledProductUrls = []
    crawledIngredientUrls = []
    ingredientsCrawled = 0
    productsCrawled = 0

    def __init__(self, crawl_num=1, *args, **kwargs):
        super(EwgSkindeepSpider, self).__init__(*args, **kwargs)
        cat = category_params1
        if int(crawl_num) is 2:
            cat = category_params2
            print("Crawler #2")
        elif int(crawl_num) is 3:
            cat = category_params3
            print("Crawler #3")
        else:
            print("Crawler #1")

        self.start_urls = [self.site_url + uri + self.req_params for uri in cat]

    def try_cast(self, value, type=int):
        # Try to cast value to input type return casted value or None if unsuccessful
        try:
            return type(value)
        except (ValueError, TypeError):
            return None

    def get_scr_bars(self, scr_brs, ldr):
        # Get values from score bar xpath, add data to input item loader
        if scr_brs:
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
            ldr.add_value('overall_hazard_score', overall_hazard_score)
            ldr.add_value('cancer_score', cancer_score)
            ldr.add_value('dev_reprod_tox_score', dev_reprod_tox_score)
            ldr.add_value('allergy_imm_tox_score', allergy_imm_tox_score)
            ldr.add_value('use_restrict_score', use_restrict_score)
        return ldr

    def get_score(self, score_img_uri):
        # Get Score from a product or ingredient page's score image
        try:
            score_img_name = score_img_uri[score_img_uri.rfind("/")+1:]
            ingredient_score = self.try_cast(
                score_img_name[:score_img_name.find(".")].replace("score_image", "")[0])
        except AttributeError:
            return None
        return ingredient_score

    def parse(self, response):
        # Parse a product group page
        self.logger.info('[parse] Called on %s', response.url)
        self.crawledCategoryUrls.append(response.url)  # Mark category page as seen

        # Find all product links in the page and add them to the list for crawling
        # Also find the product type
        # This prevents the spider from crawling the same link multiple times or looping
        found_product_links = False
        for tr in response.xpath(self.product_tr_list_xpath):
            p_type = tr.xpath(self.product_type_xpath).extract_first()
            uri = tr.xpath(self.product_link_xpath).extract_first()
            if uri and uri not in self.crawledProductUrls:
                self.crawledProductUrls.append(uri)
                product_url = urlparse.urljoin(response.url, uri)
                found_product_links = True
                self.logger.info('[parse] Found product link: %s', product_url)
                product_request = scrapy.Request(product_url, callback=self.parse_product)
                product_request.meta['product_id'] = urlsafe_b64encode(uri)
                product_request.meta['product_type'] = p_type
                yield product_request
        if not found_product_links:
            self.logger.warning('[parse] Could not extract product links from: %s', response.url)

    def parse_product(self, response):
        # Parse a product page
        self.logger.info('[parse_product] Called on %s', response.url)
        self.crawledProductUrls.append(response.url)

        # Create product item
        product_score = self.get_score(response.xpath(self.score_xpath).extract_first())
        product_ldr = ItemLoader(item=EwgScraperProduct(), response=response)
        product_ldr.add_value('product_id', response.meta['product_id'])
        product_ldr.add_value('product_type', response.meta['product_type'])
        product_ldr.add_value('url', response.url)
        product_ldr = self.get_scr_bars(response.xpath(self.score_bars_xpath), product_ldr)
        product_ldr.add_xpath('product_name', self.name_xpath)
        product_ldr.add_value('product_score', product_score)
        product_ldr.add_xpath('data_availability', self.data_availability_xpath)
        self.productsCrawled = self.productsCrawled + 1
        ingredient_list = []

        # Find then parse any ingredients on the product page
        found_ingredient_links = False
        for uri in response.xpath(self.ingredient_link_xpath).extract():
            ingredient_url = urlparse.urljoin(response.url, uri)
            ingredient_id = urlsafe_b64encode(uri)
            ingredient_list.append(ingredient_id)
            if not ingredient_url in self.crawledIngredientUrls:
                self.crawledIngredientUrls.append(uri)
                found_ingredient_links = True
                ingredient_request = scrapy.Request(ingredient_url, callback=self.parse_ingredient)
                ingredient_request.meta['ingredient_id'] = ingredient_id
                yield ingredient_request
        if not found_ingredient_links:
            self.logger.warning('[parse_product] Could not extract ingredient links from: %s', response.url)

        product_ldr.add_value('ingredient_list', ingredient_list)
        yield product_ldr.load_item()

    def parse_ingredient(self, response):
        # Parse ingredient page
        self.logger.info('[parse_ingredient] Called on %s', response.url)
        ingredient_ldr = ItemLoader(item=EwgScraperIngredient(), response=response)

        # Get ingredient score from image filename
        ingredient_score = self.get_score(response.xpath(self.score_xpath).extract_first())

        # Get score bar values
        ingredient_ldr = self.get_scr_bars(response.xpath(self.score_bars_xpath), ingredient_ldr)

        ingredient_ldr.add_value('url', response.url)
        ingredient_ldr.add_value('ingredient_id', response.meta['ingredient_id'])
        ingredient_ldr.add_xpath('ingredient_name', self.name_xpath)
        ingredient_ldr.add_value('ingredient_score', ingredient_score)
        ingredient_ldr.add_xpath('data_availability', self.data_availability_xpath)

        # Find synonym and function lists if they exists
        for other_info in response.xpath('//p[@class="tinytype"]'):
            for strong_text in other_info.xpath('strong/text()').extract():
                if strong_text and "synonym" in strong_text.lower():  # This element has the synonym list
                    synonym_list = other_info.xpath('text()').extract_first()
                    if synonym_list:
                        ingredient_ldr.add_value('synonym_list', synonym_list.split(";"))
                elif strong_text and "function" in strong_text.lower():  # This element has the functions list
                    function_list = other_info.xpath('text()').extract_first()
                    if function_list:
                        ingredient_ldr.add_value('function_list', function_list.split(";"))

        item = ingredient_ldr.load_item()
        self.crawledIngredientUrls.append(response.url)
        if "ingredient_name" in item.keys():
            self.logger.info('[parse_ingredient] Added info for %s', item['ingredient_name'])
        self.ingredientsCrawled = self.ingredientsCrawled + 1
        yield item
