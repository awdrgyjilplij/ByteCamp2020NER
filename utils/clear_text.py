import re
punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'

def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

def strQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def clear_text(line):
    line = line.strip('\n').strip()
    line = line.lower() # 变为小写
    line = re.sub(r'[\t]*', '', line) # 清除tab
    line = re.sub(r"[%s]+" %punc, "", line) # 清除标点符号
    line = strQ2B(line) # 全角转半角
    
    return line
