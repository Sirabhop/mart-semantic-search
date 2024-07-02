def clean_sentence(name):
    CHAR = [
        u'A', u'B', u'C', u'D', u'E', ' ',
        u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
        u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z',
        u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
        u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
        u'z', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
        u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
        u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
        u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
        u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
        u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'‘', u'’', u'\ufeff',
        u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9'
        ]
    return ''.join([s for s in name if s in CHAR])


def create_prompt(name_en, name_th, desc_en, desc_th):
    if (name_en != '') & (name_en is not None):
        name_en_prompt = f" or {name_en}"
    else:
        name_en_prompt = ''
    if ((desc_th != '') & (desc_th is not None)):
        desc_promt = f" have the following description {desc_th}"
        if ((desc_en != '') & (desc_en is not None)):
            desc_promt = desc_promt + f" or {desc_en}"
    else:
        if ((desc_en != '') & (desc_en is not None)):
            desc_promt =  f" have the following description {desc_en}"
        else:
            desc_promt = ''
    

    prompt = f"This product is {name_th}{name_en_prompt}{desc_promt}."

    return prompt