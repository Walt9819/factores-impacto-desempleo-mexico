import scrapy


# class JornadaSpider(scrapy.Spider):
    # name = 'jornada'
    # allowed_domains = ['www.jornada.com.mx']
    # start_urls = ['http://www.jornada.com.mx/']

    # def parse(self, response):
        # pass

from scrapy.selector import Selector
from scrapy.http.request import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import locale
from datetime import datetime
import html2text
from ..items import NewsscrapperItem


class JornadaSpider(CrawlSpider):
    name = 'jornada'
    def __init__(self, Keywords=None, *args, **kwargs):
        super(JornadaSpider, self).__init__(*args, **kwargs)
        self.allowed_domain = ['https://www.jornada.com.mx/']
        self.Keywords = Keywords
        listKeywords = "+".join(Keywords.split(','))
        self.page = 'https://www.jornada.com.mx/ultimas/@@search?SearchableText='+ listKeywords
        self.start_urls = [self.page]

    rules = {
        # Para cada item
        Rule(LinkExtractor(allow = (), restrict_xpaths = ('//div[@class="listingBar"]/span[@class="next"]/a'))),
        Rule(LinkExtractor(allow =(), restrict_xpaths = ('//div[@id="search-results"]/dl/dt')),
                            callback = 'parse_item', follow = False)
    }

    def parse_item(self, response):
        jornada_item =  NewsscrapperItem()

        jornada_item['keyWords'] = self.Keywords
        jornada_item['title'] = response.xpath('//div[@class="col-sm-12"]/h1/text()').extract()
        jornada_item['link'] =  response.request.url
        jornada_item['source'] = 'LaJornada.com'
        datefull = response.xpath('normalize-space(//div[@class="ljn-nota-datos"]/span/span[2])').extract()
        authorTemp = response.xpath('normalize-space(//div[@class="ljn-nota-datos"]/span/span[1])').extract()
        datepartial = response.xpath('normalize-space(//div[@class="article-time"]/span/text())').extract()
        #print('FECHA FORMATO 1: '+str(datefull))
        #print('FECHA FORMATO 2: '+str(datepartial))

        if len(authorTemp)>0:
            jornada_item['author'] = authorTemp[0]
            #jornada_item['date1'] = datefull
            #jornada_item['date2'] = datepartial
            #jornada_item['date'] = datetime.strptime(datefull[0],'%A, %d %b %Y %H:%M')
        else:
            jornada_item['author'] = 'Desconocido'
            #jornada_item['date1'] = 'Desconocido'
            #jornada_item['date2'] = 'Desconocido'
            # jornada_item['date'] = 'Desconocido'
            
        if datefull == '' and datepartial == '':
            jornada_item['date'] = 'Desconocido'
        elif datefull == '' and datepartial != '':
            jornada_item['date'] = datepartial
        else:
            jornada_item['date'] = datefull
            

        locationTemp = response.xpath('//div[@id="content_nitf"]/p[2]/em/text()').extract()
        if len(locationTemp) > 0:
            locationTemp = ' '.join(locationTemp)
            jornada_item['location'] = locationTemp
        else:
            locationTemp = response.xpath('//div[@id="content_nitf"]/p[1]/em/text()').extract()
            locationTemp = ' '.join(locationTemp)
            jornada_item['location'] = ''

        bodyfull = response.xpath('normalize-space(//div[@id="content_nitf"])').extract()
        parrafosText = ''
        for parrafo in bodyfull:
            parrafosText = parrafosText+ html2text.html2text(parrafo)
        #if len(parrafosText) > 5000:
        #    parrafosText = parrafosText[:5000]

        jornada_item['body'] = parrafosText
        yield jornada_item
