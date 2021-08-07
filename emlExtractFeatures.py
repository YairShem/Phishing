import re
from bs4 import BeautifulSoup


def extractSubject(a):
    # b = email.message_from_string(a)
    # bb = b['Subject']
    # return bb
    return a['Subject']


def extractBody(msg):
    body_content = ""
    if msg.is_multipart():
        for payload in msg.get_payload():
            body_content += str(payload.get_payload())
    else:
        body_content += msg.get_payload()
    return body_content


def extractReplyTo(a):
    # b = email.message_from_string(a)
    # bb = b['Reply-To']
    # return bb
    return a['Reply-To']


def extractSender(a):
    # b = email.message_from_string(a)
    # bb = b['from']
    # return bb
    return a['from']


def subjectNumChars(subject):
    return len(subject)


def subjectNumWords(subject):
    subject_Words = len(subject.split())
    return subject_Words


def bodyNumOfChars(msg):
    return len(msg)


def bodyNumOfWords(msg):
    body_noWords = len(msg.split())
    return body_noWords


def bodyNumFunctionWords(msg):
    body_noFunctionWords = 0
    wordlist = re.sub("[^A-Za-z]", " ", msg.strip()).lower().split()
    function_words = ["account", "access", "bank", "credit", "click", "identity", "inconvenience", "information",
                      "limited", "log", "minutes", "paypal", "password", "recently", "risk", "social", "security",
                      "service", "suspended", "verify"]
    for word in function_words:
        body_noFunctionWords += wordlist.count(word)
    return body_noFunctionWords


def hasWord(msg, word):
    wordlist = re.sub("[^A-Za-z]", " ", msg.strip()).lower().split()
    count = wordlist.count(word)
    if count > 0:
        return 1
    return 0


def isHtmlEmail(msg):
    if bool(BeautifulSoup(msg, "html.parser").find()):
        return 1
    return 0


def isScriptEmail(msg):
    if bool(BeautifulSoup(msg, "html.parser").find("script")):
        return 1
    return 0


def sender_replyTo(msg):
    replyTo = extractReplyTo(msg)
    if replyTo != None:
        sender = extractSender(msg)
        domainSender = re.search("@[\w.]+", sender)
        domainReplyTo = re.search("@[\w.]+", replyTo)
        if domainSender == domainReplyTo:
            return 1
        return 0
    return 0


def extractFeatures(msg):
    f = {}
    # subject = extractSubject(msg)
    # f["subjectLen"] = subjectNumChars(subject)
    # f["subjectWordsLen"] = subjectNumWords(subject)

    body = extractBody(msg)
    # f["bodyCharsNum"] = bodyNumOfChars(body)
    # f["bodyWordsNum"] = bodyNumOfWords(body)
    body_NumOfWords = bodyNumOfWords(body)
    f["bodyFuncWordsNumNormalized"] = (bodyNumFunctionWords(body) / body_NumOfWords) if (body_NumOfWords > 0) else 0

    f["bodyPaypal"] = hasWord(body, "paypal")
    f["bodyVerify"] = hasWord(body, "verify")
    f["bodyAccount"] = hasWord(body, "account")
    f["bodyAccess"] = hasWord(body, "access")
    f["hasBank"] = hasWord(body, "bank")
    f["hasCredit"] = hasWord(body, "credit")
    f["hasClick"] = hasWord(body, "click")
    f["hasIdentity"] = hasWord(body, "identity")
    f["hasInconvenience"] = hasWord(body, "inconvenience")
    f["hasInformation"] = hasWord(body, "information")
    f["hasLimited"] = hasWord(body, "limited")
    f["hasLog"] = hasWord(body, "log")
    f["hasMinutes"] = hasWord(body, "minutes")
    f["hasPassword"] = hasWord(body, "password")
    f["hasRecently"] = hasWord(body, "recently")
    f["hasRisk"] = hasWord(body, "risk")
    f["hasSocial"] = hasWord(body, "social")
    f["hasSecurity"] = hasWord(body, "security")
    f["hasService"] = hasWord(body, "service")
    f["hasSuspended"] = hasWord(body, "suspended")

    f["isHtmlEmail"] = isHtmlEmail(body)
    f["isScriptEmail"] = isScriptEmail(body)

    f["sender_replyTo"] = sender_replyTo(msg)
    return f

