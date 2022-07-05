import re
import pandas as pd
def create_categories(category):
    category = category.split("/")[0].replace(' ', '').replace(',', '&')
    return category

def convert_cat_to_number(category,dict):
    return int(dict[category])

def clean_word(word):
    return ''.join(i for i in word if i.isalpha() or i ==' ')
    return re.sub(r'[^A-Za-z0-9 ]+', '', word)

def clean_price(column):
    column = column.str.replace('£','')
    column = column.astype('str')
    column = column.str.replace(',', '')
    column = pd.to_numeric(column)
    return column