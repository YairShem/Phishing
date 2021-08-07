import socket
import re


def getLength(url):
    return len(url)


def isIpInURL(url):
    ip = r'http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.*'
    if re.search(ip, url) is not None:
        return 1
    else:
        return 0


def isMailto(url):
    if re.search('mailto:', url) is not None:
        return 1
    else:
        return 0


# Number of dots > 3 is not good!!!
def getDotsNum(url):
    return url.count('.')


# if @ appers is not good!!!
def isAtAppers(url):
    if (url.count('@') > 0):
        return 1
    else:
        return 0


def isHavingIP(url):
    x = 1
    try:
        socket.gethostbyname(url)
        x = 1
    except:
        x = 0
    return x


def getSlashNum(url):
    return url.count('/')


def getMakafNum(url):
    return url.count('-')


def isTinyUrl(url):
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                          r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                          r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                          r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                          r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                          r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                          r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                          r"tr\.im|link\.zip\.net"
    if (re.search(shortening_services, url) != None):
        return 1
    else:
        return 0


def makeStandard(url):
    if not re.match(r"^https://", url):
        if not re.match(r"^http://", url):
            return ("http://" + url)
    return url


def extractDomain(url):
    domain = re.findall(r"://([^/]+)/?", url)[0]
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")
    return domain


def subDomainsNum(url):
    l = len(re.findall("/.", url))
    return l


def extractFeatures(url):
    url2 = makeStandard(url)
    # domain = extractDomain(url)
    f = {}
    f["length"] = getLength(url2)
    f["dotsNum"] = getDotsNum(url2)
    f["isAtAppers"] = isAtAppers(url2)
    f["isHavingIP"] = isHavingIP(url2)
    f["slashNum"] = getSlashNum(url2)
    f["makafNum"] = getMakafNum(url2)
    f["isTinyUrl"] = isTinyUrl(url2)
    f["isIpInURL"] = isIpInURL(url2)
    f["isMailto"] = isMailto(url2)
    f["subDomainsNum"] = subDomainsNum(url2)
    return f

