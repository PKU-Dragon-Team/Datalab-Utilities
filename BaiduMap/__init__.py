import png, numpy
import matplotlib.pyplot as plt

import json, urllib.request, collections.abc, os, sys
from urllib.parse import quote_plus
from collections import OrderedDict

AK = None
SERVER_URL = None

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))

with open(os.path.join(__location__, 'config.json'), 'r') as config:
    x = json.load(config)
    AK = x['ak']
    SERVER_URL = x['server']

BASE_URL = "%s?ak=%s" % (SERVER_URL, AK)

class URLBuilder:
    def __init__(self, base_url):
        if '?' not in base_url:
            base_url += '?'
        self.__base_url = base_url
        self.__url = base_url
        self.__attr = OrderedDict()
    
    def __addParam(self, name, value):
        if not self.__url.endswith('&'):
           self.__url += '&'
        self.__url += "%s=%s" % (name, value)
    
    def __resetURL(self):
        self.__url = self.__base_url
    
    def addParam(self, name, value):
        self.__attr[str(name)] = str(value)
    
    def removeParam(self, name):
        try:
            del self.__attr[str(name)]
        except KeyError:
            pass
    
    def generateURL(self):
        self.__resetURL()
        for item in self.__attr.items():
            self.__addParam(item[0], item[1])
        return self.__url

def buildURL(width=None, height=None, certer=[], zoom=None, copyright=1, scale=2, bbox=[], markers=[], markerStyles=[], labels=[], labelStyles=[], paths=[], pathStyles=[]):

    url = URLBuilder(BASE_URL)
    if width:
        url.addParam('width', quote_plus(str(width)))
    if height:
        url.addParam('height', quote_plus(str(height)))
    if certer:
        if isinstance(center, str):
            url.addParam('center', quote_plus(center))
        elif isinstance(center, collections.abc.Sequence):
            url.addParam('center', quote_plus('%f,%f' % center))
    if zoom:
        url.addParam('zoom', quote_plus(str(zoom)))
    if copyright:
        url.addParam('copyright', quote_plus(str(copyright)))
    if scale:
        url.addParam('scale', quote_plus(str(scale)))
    if bbox:
        url.addParam('bbox', quote_plus('%f,%f,%f,%f' % bbox))
    if markers:
        pass
        # not implemented
    if markerStyles:
        pass
        # not implemented
    if labels:
        pass
        # not implemented
    if labelStyles:
        pass
        # not implemented
    if paths:
        pass
        # not implemented
    if pathStyles:
        pass
        # not implemented
    
    return url.generateURL()

def fetchImage(url):
    r = png.Reader(file=urllib.request.urlopen(url))
    data = r.asFloat()
    column_count = data[0]
    row_count = data[1]
    pngdata = data[2]
    plane_count = data[3]['planes']
    
    image_2d = numpy.vstack(map(numpy.float_, pngdata))
    image_3d = numpy.reshape(image_2d, (row_count, column_count, plane_count))
    
    return image_3d

def plotMap(image, alpha=1, show=False):
    # TODO: 透明、对齐
    plt.imshow(image, alpha=alpha)
    if show:
        plt.show()