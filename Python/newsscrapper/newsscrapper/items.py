# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NewsscrapperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    keyWords =  scrapy.Field()
    title = scrapy.Field()
    link = scrapy.Field()
    source = scrapy.Field()
    author = scrapy.Field()
    date = scrapy.Field()
    location = scrapy.Field()
    body = scrapy.Field()
    pass
